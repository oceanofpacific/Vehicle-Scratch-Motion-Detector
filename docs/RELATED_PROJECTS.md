# Related Projects

This project sits between several existing categories:

- general OpenCV motion detection,
- traffic and speed tracking,
- real-time AI surveillance,
- parking-lot vehicle detection,
- vehicle damage assessment after damage has already happened.

The niche here is narrower: offline review of long fixed-camera footage around one parked vehicle, with manual ROI, optional vehicle-edge and damage-zone annotations, CSV/HTML review artifacts, screenshots, clips, and candidate-only language.

## Similar Or Adjacent Repositories

| Project | What it covers | Difference from this project |
| --- | --- | --- |
| [markschnabel/opencv-motion-detector](https://github.com/markschnabel/opencv-motion-detector) | Simple Python/OpenCV background-subtraction motion detection for fixed cameras and video files. | General motion detection, without parked-vehicle ROI workflow, event report, clips, edge annotations, or behavior ranking. |
| [pageauc/speed-camera](https://github.com/pageauc/speed-camera) | Speed-camera style vehicle tracking and calibration. | Focuses on vehicle speed measurement, not scratch-risk review around a parked car. |
| [maxbeyer1/yolo-rtsp-car-detection](https://github.com/maxbeyer1/yolo-rtsp-car-detection) | YOLO and motion detection for moving vehicles in parking-lot RTSP feeds. | Real-time vehicle movement detection, not offline one-car incident triage with privacy-first reports. |
| [bappaditya-paul/AI-Surveillance-System](https://github.com/bappaditya-paul/AI-Surveillance-System) | YOLOv8, DeepSORT, human detection, behavior analysis, alerts, and dashboard. | Heavier real-time surveillance stack; this project is simpler, local, and focused on manual review candidates. |
| [artemxdata/Car-Damage-Assessment-AI](https://github.com/artemxdata/Car-Damage-Assessment-AI) | AI vehicle damage detection and assessment after visible damage is present. | Damage assessment after the fact; this project searches video for possible incident windows. |
| [andrewssobral/bgslibrary](https://github.com/andrewssobral/bgslibrary) | Broad background-subtraction library with many algorithms and Python wrappers. | Library/toolkit for moving-object detection research; this project is an end-user workflow around parked vehicle review. |

## Positioning

Vehicle Scratch Motion Detector is intentionally conservative:

- no face recognition,
- no person identification,
- no license-plate recognition,
- no crime or intent conclusion,
- local files by default,
- outputs only candidate events for manual review.
