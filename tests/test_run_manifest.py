import json
from server.run_manifest import ErrorDetail, RunManifest, RunNode, write_manifest


def read(path):
    return json.loads(path.read_text())


def test_success_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGETOOLS_INPUT_ROOT", str(tmp_path))
    m = RunManifest(runId="r1")
    m.nodes.append(RunNode(id="n1", op="save", outcome="success", artifactPath="x.png"))
    m.finish("success", final_output={"path": "x.png"})
    p = tmp_path / "manifest.json"
    write_manifest(p, m)
    d = read(p)
    assert d["status"] == "success"
    assert d["nodes"][0]["artifactPath"] == "x.png"


def test_error_manifest_retains_structured_error(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGETOOLS_INPUT_ROOT", str(tmp_path))
    err = ErrorDetail("timed out", code="TIMEOUT", retryable=True,
                      errorClass="TimeoutError", suggestion="retry")
    m = RunManifest(runId="r2")
    m.nodes.append(RunNode(id="n1", op="export", outcome="error", error=err))
    m.finish("error", error=err)
    p = tmp_path / "manifest.json"
    write_manifest(p, m)
    d = read(p)
    assert d["error"]["code"] == "TIMEOUT"
    assert d["error"]["retryable"] is True
    assert d["nodes"][0]["error"]["errorClass"] == "TimeoutError"


def test_partial_manifest_keeps_success_and_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGETOOLS_INPUT_ROOT", str(tmp_path))
    m = RunManifest(runId="r3")
    m.nodes += [
        RunNode(id="n1", op="a", outcome="success"),
        RunNode(id="n2", op="b", outcome="error", error=ErrorDetail("bad")),
    ]
    m.finish("partial")
    p = tmp_path / "manifest.json"
    write_manifest(p, m)
    d = read(p)
    assert d["status"] == "partial"
    assert [n["outcome"] for n in d["nodes"]] == ["success", "error"]


def test_in_progress_manifest_is_atomic_json(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGETOOLS_INPUT_ROOT", str(tmp_path))
    p = tmp_path / ".runs" / "r4" / "manifest.json"
    write_manifest(p, RunManifest(runId="r4"))
    assert read(p)["status"] == "in_progress"
    assert not list(p.parent.glob(".manifest.json.*.tmp"))
