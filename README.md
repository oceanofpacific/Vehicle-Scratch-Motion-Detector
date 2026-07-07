# Vehicle Scratch Motion Detector

Vehicle Scratch Motion Detector is a local Python + OpenCV tool for reviewing long fixed-camera surveillance videos around a parked vehicle. It helps surface candidate time ranges where motion, trajectories, or unusual near-vehicle behavior may be worth manual review.

Important: this project only produces review candidates. It does not identify people, infer intent, assign responsibility, or conclude that a crime happened.

## What It Does

- Accepts one video, multiple videos, a folder, or glob patterns.
- Lets you select a vehicle ROI from the first frame.
- Analyzes only the ROI for the basic motion workflow.
- Uses OpenCV background subtraction, denoising, thresholding, morphology, and contours.
- Merges nearby motion events to avoid duplicate clips for one passerby.
- Writes `events.csv`, `report.html`, screenshots, optional event clips, and an optional debug video.
- Supports preview and calibration modes for tuning parameters.
- Supports vehicle-edge annotation, damage-zone annotation, trajectory screening, and near-vehicle dwell/slow-speed behavior screening.
- Runs locally on macOS and Windows.

## Installation

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Optional: install `ffmpeg` and make sure it is available on `PATH`. If `ffmpeg` is missing, the tool skips clip export without stopping the scan.

## Basic Motion Scan

```bash
python main.py --input videos/input.mp4 --output output --min-area 800 --min-duration 1.5 --merge-gap 10 --pre-roll 15 --post-roll 15 --sample-every 2
```

If `--roi x,y,w,h` is not provided, the program opens the first frame and asks you to select the vehicle plus roughly 1-2 meters around it. Press `Enter` or `Space` to confirm the ROI.

Outputs:

```text
output/
  events.csv
  report.html
  screenshots/
  clips/
  debug_video.mp4
```

## Preview Mode

Use preview mode while tuning parameters. It shows the ROI, mask, motion boxes, and timestamp.

```bash
python main.py --input videos/input.mp4 --preview --min-area 800 --sample-every 2
```

## Calibration Mode

Calibration mode samples several random time points and saves preview images so you can compare different `--min-area`, `--min-duration`, and `--sample-every` values.

```bash
python main.py --input videos/input.mp4 --output output --calibration --calibration-samples 12 --min-area 800 --sample-every 2
```

Outputs:

```text
output/
  calibration/
```

## Vehicle Edge And Damage Zones

For a more focused review, annotate the visible vehicle body edge:

```bash
python main.py --input videos/input.mp4 --output output_edge --annotate-vehicle-edge
```

Then run trajectory screening near the annotated edge:

```bash
python main.py --input videos/input.mp4 --output output_edge_scan --vehicle-edge-file output_edge/vehicle_edges.json --trajectory-scan --edge-distance 60 --min-area 800 --min-duration 1.5 --merge-gap 10 --sample-every 10
```

If you know likely damage locations, annotate key zones such as left rear or right side:

```bash
python main.py --input videos/input.mp4 --output output_edge --annotate-damage-zones --damage-zone-names left_rear,right_side
```

Then keep only trajectory events that approach those marked zones:

```bash
python main.py --input videos/input.mp4 --output output_damage_scan --vehicle-edge-file output_edge/vehicle_edges.json --damage-zones-file output_edge/damage_zones.json --trajectory-scan --require-damage-zone --edge-distance 60 --damage-distance 80 --sample-every 10
```

## Behavior Scan

When direct contact is not obvious, behavior scan ranks tracked moving targets that:

- stay near the vehicle longer than other targets,
- move much slower near the vehicle than elsewhere in the frame,
- approach marked damage zones.

```bash
python main.py --input videos/input.mp4 --output output_behavior --roi 100,100,800,500 --vehicle-edge-file output_edge/vehicle_edges.json --damage-zones-file output_edge/damage_zones.json --behavior-scan --min-area 800 --sample-every 10 --behavior-analysis-padding 220 --behavior-scale 0.5 --behavior-min-near-duration 2 --behavior-top 80
```

Outputs:

```text
output_behavior/
  behavior_events.csv
  behavior_report.html
  behavior_screenshots/
```

## CSV Fields

`events.csv` includes:

- `event_id`
- `start_time`
- `end_time`
- `start_timestamp_hhmmss`
- `end_timestamp_hhmmss`
- `duration_seconds`
- `video_filename`
- `frame_start`
- `frame_end`
- `max_motion_area`
- `confidence_score`
- `screenshot_path`
- `clip_path`

`confidence_score` is rule-based, not machine learning. It increases with longer duration, larger motion area, proximity to the vehicle edge, repeated activity in a similar area, and proximity to marked damage zones.

## Common Arguments

| Argument | Default | Description |
| --- | ---: | --- |
| `--input` | required | Video file, directory, or glob pattern. Can be repeated. |
| `--output` | `output` | Output directory. |
| `--roi` | empty | ROI as `x,y,w,h`. If omitted, select it with the mouse. |
| `--min-area` | `800` | Minimum foreground contour area in pixels. |
| `--min-duration` | `1.5` | Minimum continuous motion duration in seconds. |
| `--merge-gap` | `10` | Merge events separated by less than this many seconds. |
| `--pre-roll` | `15` | Seconds before each event to include in exported clips. |
| `--post-roll` | `15` | Seconds after each event to include in exported clips. |
| `--sample-every` | `2` | Analyze every Nth frame for speed. |
| `--preview` | off | Show interactive ROI, mask, and motion boxes. |
| `--calibration` | off | Save random sampled detection previews. |
| `--no-debug-video` | off | Skip `debug_video.mp4`. |
| `--annotate-vehicle-edge` | off | Click visible vehicle edge points and save JSON. |
| `--annotate-damage-zones` | off | Draw damage-prone zones and save JSON. |
| `--trajectory-scan` | off | Track motion near the annotated vehicle edge. |
| `--behavior-scan` | off | Rank long-dwell or slow-near-vehicle behavior candidates. |

Run `python main.py --help` for the full argument list.

## Privacy And Responsible Use

- Keep original surveillance videos local unless you have permission to share them.
- Review and redact faces, license plates, addresses, timestamps, and private locations before publishing screenshots or clips.
- Treat all outputs as candidate evidence for human review only.
- False positives can be caused by shadows, headlights, rain, snow, reflections, trees, camera shake, compression artifacts, and occlusion.

See [PRIVACY.md](PRIVACY.md) for more guidance.

## Limitations

- The tool uses classical computer vision and simple tracking. It is meant for triage, not proof.
- Pixel speed is only useful for comparing motion inside the same fixed camera view.
- Night footage, low frame rates, heavy compression, and strong lighting changes may need parameter tuning.
- The project does not perform face recognition, person identification, license plate recognition, or criminal judgment.

## License

MIT License. See [LICENSE](LICENSE).
