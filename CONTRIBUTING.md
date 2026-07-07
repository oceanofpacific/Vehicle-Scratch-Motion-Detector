# Contributing

Thank you for helping improve this project.

## Good Contributions

- Bug fixes for Windows, macOS, and OpenCV compatibility.
- Clear parameter tuning examples.
- Better false-positive reduction.
- Documentation improvements.
- Tests or small reproducible examples that do not include private surveillance footage.

## Privacy Rules

Please do not attach private surveillance videos, unredacted faces, license plates, addresses, or exact private camera locations to public issues or pull requests.

If a visual example is necessary, use synthetic footage, public-domain footage, or heavily redacted screenshots.

## Development

```bash
python -m venv .venv
pip install -r requirements.txt
python -m py_compile main.py vehicle_motion_detector.py
```

Before opening a pull request, run the tool on a short local test clip if you have one available.
