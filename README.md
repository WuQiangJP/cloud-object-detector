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

## Local setup

```bash
git clone https://github.com/Username/cloud-object-detector.git
cd cloud-object-detector

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.in
# Or use the locked versions:
# python -m pip install -r requirements.lock.txt

python -m streamlit run app.py
```

## Troubleshooting

`ImportError: libGL.so.1: cannot open shared object file.`

This happens when `opencv-python` is installed in a headless Linux
environment (Docker, GitHub Codespaces, servers). Use the official
`ultralytics-opencv-headless` package instead of `ultralytics`.

```bash
pip uninstall -y ultralytics opencv-python
pip install ultralytics-opencv-headless
```

## License

This project is provided for learning and research purposes.

It depends on the Ultralytics YOLO package, which is distributed under
AGPL-3.0. Review the upstream license before commercial use or before
offering this application as a public network service.
