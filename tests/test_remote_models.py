"""Model-server options: ASR words, vision frames and detection boxes from an
OpenAI-compatible server such as LocalAI, so the image needs no local models."""
import json

import httpx

import gemini_worker
import llm_backend
import transcribe_backends


def test_segments_split_on_pauses_and_length():
    words = [{"word": " one", "start": 0.0, "end": 0.4},
             {"word": " two", "start": 0.5, "end": 0.9},
             {"word": " three", "start": 2.0, "end": 2.4}]
    segments = transcribe_backends._segments_from_words(words)
    assert [(s["start"], s["end"], s["text"]) for s in segments] == [
        (0.0, 0.9, " one two"), (2.0, 2.4, " three")]


def test_vision_frames_go_to_the_vision_server_as_images(monkeypatch):
    monkeypatch.setenv("VISION_BASE_URL", "http://vision.test/v1")
    monkeypatch.setenv("VISION_MODEL", "gemma-3-4b-it")
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        answer = {"layout": "split", "confidence": 0.8, "why": "two people"}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(answer)}}]})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(llm_backend, "_client", lambda **kw: httpx.Client(transport=transport, **kw))

    answer = llm_backend.generate_vision_json("pick", [b"jpg"], gemini_worker.LayoutChoice)

    assert answer["layout"] == "split"
    assert seen["url"] == "http://vision.test/v1/chat/completions"
    assert seen["body"]["model"] == "gemma-3-4b-it"
    parts = seen["body"]["messages"][1]["content"]
    assert parts[0]["image_url"]["url"] == "data:image/jpeg;base64,anBn"
    assert parts[1] == {"type": "text", "text": "pick"}
