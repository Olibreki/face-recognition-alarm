# Embedded Facial-Recognition Alarm System

Real-time facial-recognition alarm running on a Raspberry Pi. A Python capture pipeline grabs frames from a camera, detects faces, and hands them off over HTTP to a custom C inference service that runs face-embedding matching against known reference faces and fires a Telegram alert when an unrecognized face is detected.

## How it works

```
Camera --> Python capture pipeline --> HTTP --> C inference service --> Telegram alert
           (Flask, Haar cascade,                (ONNX Runtime C API,
            MTCNN)                               libmicrohttpd, libcurl)
```

1. **Capture** - a Python process (Flask) pulls frames from the Pi camera, uses a Haar cascade / MTCNN to locate faces, and crops/streams detected face regions.
2. **Inference** - a C service loads a fine-tuned ONNX face-embedding model (InsightFace buffalo_l, adapted via last-layer transfer learning) through the ONNX Runtime C API, computes an embedding for each incoming face, and compares it against known reference embeddings using cosine similarity.
3. **Alerting** - on an unrecognized face (or a configured trigger condition), the service sends a Telegram alert via libcurl.

## Tech stack

- **Python** - Flask, OpenCV (Haar cascade), MTCNN
- **C** - ONNX Runtime C API, libmicrohttpd (HTTP server), libcurl (Telegram alerting)
- **Model** - ONNX face-embedding model (InsightFace buffalo_l), fine-tuned via last-layer transfer learning
- **Platform** - Raspberry Pi

## Repository structure

```
.
├── live_feed_detect_face.py   # Python capture + face-detection pipeline
├── live_feed_detect_face.c    # C live-feed entry point
├── face_recognition_alarm.c   # C inference service - ONNX matching + Telegram alerting
└── include/                   # ONNX Runtime C API headers
```

Build artifacts, the Python virtual environment, the ONNX Runtime binary distribution, and any captured/training photos or trained model weights are intentionally excluded from version control - see .gitignore.

## Setup

### Requirements

- Raspberry Pi (tested on Raspberry Pi OS, aarch64)
- Python 3.11+
- ONNX Runtime (Linux ARM64 build)
- A Telegram bot token + chat ID for alerting

### Python environment

```bash
python3 -m venv LVPython --system-site-packages
source LVPython/bin/activate
pip install -r requirements.txt
```

### C build

The C services link against the ONNX Runtime C API, libmicrohttpd, and libcurl. Download the ONNX Runtime Linux ARM64 release for your platform (the include/ folder in this repo already has the headers) and point the build at its lib/ directory, then compile, e.g.:

```bash
gcc face_recognition_alarm.c -Iinclude -L<onnxruntime>/lib -lonnxruntime -lmicrohttpd -lcurl -o face_recognition_alarm
```

### Environment variables

Configuration (Telegram credentials, camera source, model/embedding paths, etc.) is supplied via environment variables - see .env.example for the full list. Copy it to .env and fill in your own values; .env is gitignored and should never be committed.

## Notes

This repo intentionally does not include the fine-tuned model weights, reference face embeddings, or any captured/training photos, since these contain personal biometric data. To run the system yourself, you'll need to fine-tune your own embedding model and generate your own reference embeddings from your own reference photos.
