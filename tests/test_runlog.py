from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpustacker import cli
from gpustacker.batch import FrameGroup, GroupKey, run_batch
from gpustacker.pipeline import PipelineSettings
from gpustacker.runlog import RunLog
from gpustacker.stacking import FrameStore


def test_run_log_appends_messages_and_exception_traceback(tmp_path: Path) -> None:
    path = tmp_path / "stack.log"

    with RunLog(path) as run_log:
        run_log.write("Registering frame 1")
        try:
            raise OSError("write failed")
        except OSError as exc:
            run_log.write_exception("Run failed", exc)

    with RunLog(path) as run_log:
        run_log.write("Retry started")

    text = path.read_text(encoding="utf-8")
    assert "Registering frame 1" in text
    assert "Run failed: OSError: write failed" in text
    assert "raise OSError(\"write failed\")" in text
    assert "Retry started" in text
    assert text.count("Run started") == 2


def test_cli_logs_pipeline_write_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "master.fit"

    class FailingPipeline:
        def __init__(self, settings, status, progress):
            self.status = status

        def run(self):
            self.status("Registering frame 1")
            raise OSError("simulated write failure")

    monkeypatch.setattr(cli, "settings_from_args", lambda args: SimpleNamespace(output=output))
    monkeypatch.setattr(cli, "StackingPipeline", FailingPipeline)

    with pytest.raises(OSError, match="simulated write failure"):
        cli.main(["stack", "input.fit", "--output", str(output)])

    text = output.with_suffix(".log").read_text(encoding="utf-8")
    assert "Registering frame 1" in text
    assert "Run failed: OSError: simulated write failure" in text
    assert "Traceback (most recent call last)" in text


def test_async_frame_write_error_identifies_source_and_store_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_save(*args, **kwargs):
        raise OSError("80545752 requested and 0 written")

    monkeypatch.setattr("gpustacker.stacking.np.save", fail_save)
    store = FrameStore(tmp_path / "registered", (1, 2, 2))

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = store.add_async("frame-17.fit", np.zeros((1, 2, 2), dtype=np.float32), None, executor)
        with pytest.raises(OSError, match=r"frame-17\.fit.*reg_0000\.npy.*16 bytes"):
            future.result()


def test_batch_failure_status_contains_traceback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingPipeline:
        def __init__(self, settings, status, progress):
            pass

        def run(self):
            raise OSError("simulated batch write failure")

    monkeypatch.setattr("gpustacker.pipeline.StackingPipeline", FailingPipeline)
    lights = [tmp_path / "one.fit", tmp_path / "two.fit"]
    settings = PipelineSettings(lights=lights, output=tmp_path / "master.fit")
    group = FrameGroup(GroupKey((("FILTER", "LP"),)), lights)
    messages: list[str] = []

    result = run_batch(settings, [group], tmp_path, status_cb=messages.append)

    assert result.outputs[0][1] is None
    failure = next(message for message in messages if message.startswith("Group failed:"))
    assert "OSError: simulated batch write failure" in failure
    assert "Traceback (most recent call last)" in failure