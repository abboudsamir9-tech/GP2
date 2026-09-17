# ASL Stereo Recognition

Contract-driven implementation of real-time ASL recognition using synchronized
front and side cameras, 3D landmark fusion, temporal classification, and a
desktop UI.

Sprint 0/1 currently provides immutable stage contracts and pure preprocessing
math. Camera, MediaPipe, model-training, runtime, and UI modules are scaffolded
but intentionally not implemented yet.

Live runtime frames must remain in volatile memory and must never be persisted
or transmitted.

