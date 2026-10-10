"""A job that dies mid-render must come back with its own settings, and a job that gave up
must not be reported as finished with clips that were never written."""
import json

import pytest

app_module = pytest.importorskip("app")


def test_job_settings_keep_what_the_job_set_and_drop_secrets(monkeypatch):
    monkeypatch.delenv("SCREENCAST_LAYOUT", raising=False)
    monkeypatch.setenv("PATH", "/usr/bin")
    env = {"SCREENCAST_LAYOUT": "1", "CLIP_TARGET_MAX": "2", "LLM_API_KEY": "x", "PATH": "/usr/bin"}
    assert app_module.job_settings(env) == {"SCREENCAST_LAYOUT": "1", "CLIP_TARGET_MAX": "2"}


def test_recovery_lists_only_written_clips_and_fails_a_job_that_gave_up(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(app_module, "jobs", {})
    job = tmp_path / "j1"
    job.mkdir()
    (job / "v_metadata.json").write_text(json.dumps({"shorts": [
        {"video_url": "/videos/j1/v_clip_1.mp4"}, {"video_url": "/videos/j1/v_clip_2.mp4"}]}))
    (job / "v_clip_1.mp4").write_bytes(b"x")
    (job / ".gave_up").touch()

    app_module._recover_jobs_from_disk()

    recovered = app_module.jobs["j1"]
    assert recovered["status"] == "failed"
    assert [c["video_url"] for c in recovered["result"]["clips"]] == ["/videos/j1/v_clip_1.mp4"]
