# MotionCrop

Automatic aircraft video stabilization and cropping tool.

**Author:** Jim Bourke
**Copyright:** 2017
**License:** [GNU General Public License v3](LICENSE)

> Unless you are a member of a world aerobatic team, in which case your use of this software is conditional on your agreement that the US team is the best. :D

## What Does It Do?

Given a video of an airplane surrounded by sky, it crops the video around the airplane to keep it centered in frame.

![MotionCrop example output](docs/screenshot.png)

*Example output from the included sample video, showing the aircraft kept centered in the cropped frame.*

## How It Works

1. Each frame is converted from RGB to HSV color space
2. Hue and Saturation are discarded, keeping only Value
3. Otsu threshold is calculated for the frame
4. Canny edge detection is applied using the Otsu threshold
5. Dilation operator closes the detected edges
6. Edges are converted to contours
7. The contour with the largest area is selected as the aircraft
8. Center of mass (centroid) is calculated from the contour's moments
9. Original frame is cropped around the calculated center
10. Cropped frame is written to output file

## Installation

Download the following files and place them in a folder:
- `motioncrop.exe`
- `opencv_ffmpeg320_64.dll`
- `opencv_world320.dll`

Add the folder to your system PATH.

> **Platform note:** MotionCrop is currently packaged and distributed as a Windows application, and the provided binaries and build instructions target Windows. The core code is standard C++ and OpenCV, so it should compile on Linux or macOS with an appropriate build system and OpenCV installation, but no cross‑platform build files are included yet.

## Usage

You can run MotionCrop either with positional arguments (original style) or with named flags.

**Positional usage (backwards compatible):**

```
motioncrop filename [windowSize] [threshold] [iterations] [verbose] [-nogui]
```

**Flag-based usage:**

```
motioncrop filename [-w windowSize] [-t threshold] [-i iterations] [-v [level]] [-nogui]
```

| Option / Position | Default | Range    | Description                                      |
|-------------------|---------|----------|--------------------------------------------------|
| `filename`        | (req.)  | -        | Input video file                                 |
| `windowSize` / `-w`, `-window` | 400     | 50–5000  | Output crop size in pixels                       |
| `threshold` / `-t`, `-threshold` | 1.0   | 0.05–2.0 | Edge detection threshold scalar; try 0.5 if jittery |
| `iterations` / `-i`, `-iterations` | 2   | 1–5      | Dilation iterations; higher can reduce jitter    |
| `verbose` / `-v`, `-verbose` | 0       | 0–1      | Debug mode (extra windows and console output)    |
| `-nogui`           | off     | -        | Disable all OpenCV windows (batch/non‑interactive mode) |

Output is saved as `{filename}_mcrop.avi` using the DivX codec.

### Example

Run the included sample from the `samples/` directory:
```
cd samples
sample.bat
```

Or directly:
```
motioncrop samples/sample.MP4
```

There is only one sample file in this repository. This is intentional: GitHub's repository size limits make it impractical to ship many large videos, and this particular sample was chosen to highlight both cases where MotionCrop works well and where its behavior breaks down.

## Limitations

The algorithm works best with dark, uniformly colored aircraft against clear sky. It may produce poor results with:

- Ground visible in frame
- Sun or bright light sources
- Birds or other objects between camera and aircraft
- Out of focus footage
- Airshow smoke
- Harsh cloud edges
- Heavy video compression artifacts

Frame-to-frame jitter can occur, usually caused by sun glints.

## Troubleshooting

**Error when running:** The program requires the DivX codec. Install it if not present on your system.

## Building from Source

### Prerequisites

- Visual Studio 2017 or later
- [vcpkg](https://github.com/microsoft/vcpkg) package manager

### Setup vcpkg (one-time)

```bash
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.bat
./vcpkg integrate install
```

### Build

vcpkg will automatically install OpenCV dependencies on first build:

```bash
msbuild "src/MotionCrop.sln" -p:Configuration=Release -p:Platform=x64
```

The first build takes longer as vcpkg downloads and compiles OpenCV. The resulting `motionCrop.exe` will be placed in `src/x64/Release`, which is the path expected by the test script in this repository.

### Tests

To validate argument parsing and exit codes, you can run the shell test script after building:

```bash
bash tests/test_args.sh
```

On Windows, run this from WSL or another Bash environment. On Linux or macOS, adjust the `EXE` path in `tests/test_args.sh` if your build output location differs from `src/x64/Release/motionCrop.exe`.
