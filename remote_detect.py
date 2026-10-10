"""Face and person boxes from a LocalAI-style /v1/detection endpoint.

Set DETECT_BASE_URL (e.g. http://localai:8080/v1) to run detection on that
server, with DETECT_FACE_MODEL and DETECT_PERSON_MODEL naming its models.
The image then needs neither MediaPipe nor YOLO.
"""
import base64
import os

import cv2
import httpx

FACE_MODEL = os.environ.get("DETECT_FACE_MODEL", "face-detect-buffalo-l")
PERSON_MODEL = os.environ.get("DETECT_PERSON_MODEL", "rfdetr-cpp-medium")

_client = None


def enabled():
    return bool(os.environ.get("DETECT_BASE_URL"))


def detect(frame, model):
    """Detections on a BGR frame as dicts with x, y, width, height (pixels), class_name, confidence."""
    global _client
    if _client is None:
        _client = httpx.Client(
            base_url=os.environ["DETECT_BASE_URL"].rstrip("/"),
            headers={"Authorization": f"Bearer {os.environ.get('DETECT_API_KEY', 'none')}"},
            timeout=float(os.environ.get("DETECT_TIMEOUT", "60")))
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return []
    image = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()
    resp = _client.post("/detection", json={"model": model, "image": image})
    resp.raise_for_status()
    return resp.json().get("detections") or []


def faces(frame):
    return detect(frame, FACE_MODEL)


def people(frame):
    return [d for d in detect(frame, PERSON_MODEL) if d.get("class_name") == "person"]
