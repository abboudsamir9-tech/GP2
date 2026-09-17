# AGENTS.md - ASL Stereo Recognition Pipeline

## Core Architecture & Contracts
- Landmarks: Exactly 46 joints (21 per hand, 4 upper-body pose joints: shoulders & elbows). Discard all face mesh and lower-body joints[cite: 1, 2].
- Features: 138-dimensional float32 vector per frame (46 × 3)[cite: 1, 2].
- Preprocessing: Translate origin to mid-shoulder (0, 0, 0) and scale by inter-shoulder Euclidean distance[cite: 1, 2].
- Temporal Buffer: Tensor shape (1, 45, 138) with stride 8[cite: 1, 2]. Skip incrementing on dropped frames.
- Camera Backend: Windows requires `cv2.CAP_DSHOW` and a 5.0s warmup timeout[cite: 1]. Support single-camera fallback when only 1 webcam is connected[cite: 1].
- Privacy: All frames processed in volatile RAM only; never write raw video frames to disk.

## Environment
- Python 3.11 (`venv311`)[cite: 1]
- PyTorch: CPU-only (`torch==2.3.0+cpu`)[cite: 1]
- Dependencies: `numpy<2.0.0,>=1.26.0`, `mediapipe==0.10.14`, `PyQt5==5.15.10`[cite: 1]. MediaPipe 0.10.14 preserves the identical strict 46-joint extraction contract while avoiding the documented Windows C++ DLL initialization crash (`_framework_bindings`) observed with 0.10.21.
