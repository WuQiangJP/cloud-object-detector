# Cloud Object Detector

A beginner-friendly object detection demo running in GitHub Codespaces.

## Features

- Browser-based Streamlit interface
- CPU inference in GitHub Codespaces
- Pretrained Ultralytics YOLO model
- Image upload with size and pixel limits
- No application-level saving of uploaded images

## Privacy notice

Uploaded images are transferred to the current GitHub Codespace for inference.

Do not upload private photos, identity documents, unpublished research data,
or sensitive remote sensing imagery.

The default Codespaces forwarded port should remain private for personal testing.

## Dependency notice

This project uses the Ultralytics Python package.
Review the Ultralytics AGPL-3.0 license before commercial redistribution
or offering this application as a public network service.

## Troubleshooting

`ImportError: libGL.so.1: cannot open shared object file.`

This happens when `opencv-python` is installed in a headless Linux
environment (Docker, GitHub Codespaces, servers). Use the official
`ultralytics-opencv-headless` package instead of `ultralytics`.

```bash
pip uninstall -y ultralytics opencv-python
pip install ultralytics-opencv-headless
