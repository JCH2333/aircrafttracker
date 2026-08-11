"""Optional SAM 2.1 video mask provider."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stabilize.tracking.models import BBox


class Sam2Unavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class Sam2Prompt:
    frame_idx: int
    bbox: BBox
    source: str = "auto"
    score: float = 1.0


@dataclass
class Sam2Mask:
    frame_idx: int
    mask: np.ndarray
    quality: float


class Sam2MaskProvider:
    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = True,
    ):
        self.model_id = model_id
        self.device = device
        self.offload_video_to_cpu = offload_video_to_cpu
        self.offload_state_to_cpu = offload_state_to_cpu
        self._predictor = None

    @staticmethod
    def availability() -> tuple[bool, str]:
        try:
            import torch
        except ImportError:
            return False, "PyTorch is not installed"
        if not torch.cuda.is_available():
            return False, "CUDA-enabled PyTorch is not available"
        try:
            import sam2  # noqa: F401
        except ImportError:
            return False, "SAM 2 is not installed"
        return True, ""

    def load(self) -> None:
        available, reason = self.availability()
        if not available:
            raise Sam2Unavailable(reason)
        try:
            from sam2.sam2_video_predictor import SAM2VideoPredictor

            self._predictor = SAM2VideoPredictor.from_pretrained(
                self.model_id,
                device=self.device,
            )
        except Exception as exc:
            raise Sam2Unavailable(f"Could not load SAM 2 model: {exc}") from exc

    def propagate(
        self,
        frame_dir: Path | str,
        prompts: list[Sam2Prompt],
    ):
        if not prompts:
            raise ValueError("SAM 2 requires at least one box prompt")
        if self._predictor is None:
            self.load()

        prompts = sorted(prompts, key=lambda prompt: prompt.frame_idx)
        earliest = prompts[0].frame_idx

        if earliest > 0:
            reverse_state = self._create_state(frame_dir)
            self._add_prompts(reverse_state, prompts)
            reverse_masks = {}
            for frame_idx, _, logits in self._predictor.propagate_in_video(
                reverse_state,
                start_frame_idx=earliest,
                max_frame_num_to_track=earliest + 1,
                reverse=True,
            ):
                if frame_idx != earliest:
                    reverse_masks[frame_idx] = self._convert_mask(frame_idx, logits)
            for frame_idx in sorted(reverse_masks):
                yield reverse_masks[frame_idx]

        state = self._create_state(frame_dir)
        self._add_prompts(state, prompts)
        for frame_idx, _, logits in self._predictor.propagate_in_video(
            state,
            start_frame_idx=earliest,
            reverse=False,
        ):
            yield self._convert_mask(frame_idx, logits)

    def _create_state(self, frame_dir: Path | str):
        return self._predictor.init_state(
            video_path=str(frame_dir),
            offload_video_to_cpu=self.offload_video_to_cpu,
            offload_state_to_cpu=self.offload_state_to_cpu,
            async_loading_frames=True,
        )

    def _add_prompts(self, state, prompts: list[Sam2Prompt]) -> None:
        for prompt in prompts:
            x, y, w, h = prompt.bbox
            box = np.array([x, y, x + w, y + h], dtype=np.float32)
            self._predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=prompt.frame_idx,
                obj_id=1,
                box=box,
            )

    @staticmethod
    def _convert_mask(frame_idx: int, logits) -> Sam2Mask:
        import torch

        object_logits = logits[0]
        mask_tensor = object_logits > 0.0
        if bool(mask_tensor.any()):
            margin = torch.sigmoid(torch.abs(object_logits[mask_tensor]))
            quality = float(((margin.mean() - 0.5) * 2.0).clamp(0, 1).item())
        else:
            quality = 0.0
        mask = mask_tensor.detach().cpu().numpy().squeeze().astype(bool)
        return Sam2Mask(frame_idx=frame_idx, mask=mask, quality=quality)
