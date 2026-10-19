# cloud-object-detector

## Troubleshooting

`ImportError: libGL.so.1: cannot open shared object file`

This happens when `opencv-python` is installed in a headless Linux
environment (Docker, GitHub Codespaces, servers). Use the official
`ultralytics-opencv-headless` package instead of `ultralytics`.

```bash
pip uninstall -y ultralytics opencv-python
pip install ultralytics-opencv-headless