"""UI-safe mappings for pipeline telemetry and prediction records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class TelemetryDisplay:
    sync_text: str
    sync_ok: bool
    tracking_text: str
    tracking_ok: bool
    fusion_text: str
    fusion_ok: bool
    fps_text: str
    confidence_percent: int


def map_telemetry(
    telemetry: Mapping[str, Any], *, confidence: float = 0.0
) -> TelemetryDisplay:
    delta_ms = abs(float(telemetry.get("sync_delta_ms", float("inf"))))
    single_camera = bool(telemetry.get("single_camera", False))
    sync_ok = not single_camera and delta_ms <= 40.0
    tracking = bool(telemetry.get("mediapipe_tracking", False))
    fusion_mode = str(telemetry.get("fusion_status", "Fallback"))
    fusion_ok = fusion_mode.casefold() == "triangulated"
    fps = max(0.0, float(telemetry.get("fps", 0.0)))
    confidence_percent = round(max(0.0, min(1.0, confidence)) * 100)
    return TelemetryDisplay(
        sync_text=(
            "[SINGLE-CAMERA FALLBACK]"
            if single_camera
            else "Synchronized" if sync_ok else "Desynced"
        ),
        sync_ok=sync_ok,
        tracking_text="Active" if tracking else "Lost",
        tracking_ok=tracking,
        fusion_text=(
            "[SINGLE-CAMERA FALLBACK]"
            if single_camera
            else "Triangulated" if fusion_ok else "Fallback"
        ),
        fusion_ok=fusion_ok,
        fps_text=f"{fps:.1f} FPS",
        confidence_percent=confidence_percent,
    )


def prediction_text(gloss: str, confidence: float, latency_ms: float) -> str:
    safe_gloss = " ".join(str(gloss).strip().split())
    return f"{safe_gloss} · {confidence:.0%} · {latency_ms:.1f} ms"


__all__ = ["TelemetryDisplay", "map_telemetry", "prediction_text"]
