# VisionShift AI
Runnable prototype: WebGL live filters, optional trained neural style worker, recording and manual WebRTC sender signaling. Camera frames remain local unless AI style is selected. Use only with consent.

## Run
`python -m http.server 8080 --directory web` from repository root. Open http://localhost:8080. Grant camera access. Mobile requires HTTPS. Desktop: `cd desktop && npm install && npm start`.

## Worker
`cd worker && pip install -r requirements.txt` then `python train.py source_images target_images --epochs 10` (matching filenames, paired examples). `uvicorn main:app --port 8000`. Training produces weights and ONNX export. No pretrained weights supplied. Docker: `docker build -t visionshift-worker worker` and `docker run -p 8000:8000 -v "$PWD/worker/style.pt:/app/style.pt:ro" visionshift-worker`. For GPU use a CUDA-compatible PyTorch environment and NVIDIA container runtime. Set ORIGINS to the exact frontend origin. Remote worker needs HTTPS, authentication and rate limits before public exposure.

## Output
Record saves transformed video without audio. WebRTC offer requires a compatible receiving peer and manual SDP answer exchange; no signaling server or TURN is included. OBS can capture the desktop window and expose OBS Virtual Camera. No native virtual camera driver is included.

## Deployment
Import root into Vercel as a static project. Deploy worker separately; do not package PyTorch into Vercel functions. Set worker URL in UI.

## Tests
`pytest tests` with worker dependencies installed. Browser camera, recording, GPU and device compatibility require manual testing.

## Scope limitations
This is a starter implementation, not a complete production application. Face tracking, segmentation, avatar animation, prompt-conditioned diffusion, native mobile builds, RTMP and native webcam drivers are not implemented. Neural style model is a small supervised network, not a generative diffusion model. No performance benchmarks or real camera verification have been performed.
