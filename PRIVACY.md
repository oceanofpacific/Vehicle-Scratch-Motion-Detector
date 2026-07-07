# Privacy And Responsible Use

This project is designed for local review of fixed-camera surveillance footage. Surveillance footage can contain sensitive information, including faces, license plates, homes, workplaces, timestamps, and location details.

## Before Sharing Anything

- Do not upload original surveillance videos unless you have the right to share them.
- Review screenshots and clips before sharing them.
- Redact or crop faces, license plates, house numbers, addresses, account names, and private locations.
- Avoid publishing exact camera locations, file names, timestamps, or annotations that identify a private scene.
- Remember that `vehicle_edges.json` and `damage_zones.json` can reveal private camera geometry and vehicle position.

## What The Tool Does Not Do

- It does not identify a person.
- It does not infer intent.
- It does not determine responsibility.
- It does not conclude that a crime happened.
- It does not replace human review or legal advice.

## Interpreting Results

The generated CSV, HTML report, screenshots, clips, and behavior scores are only triage aids. They can help you decide which time ranges deserve closer review, but false positives and missed events are expected.

Common false-positive sources include shadows, headlights, reflections, rain, snow, trees, compression artifacts, camera shake, and overlapping people or vehicles.

## Recommended Public Issue Policy

If you open a public GitHub issue, avoid attaching private footage. Prefer:

- a short description of the problem,
- OpenCV and Python versions,
- the command you ran,
- redacted screenshots,
- synthetic or public-domain sample videos.
