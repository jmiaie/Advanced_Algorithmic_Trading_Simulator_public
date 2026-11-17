"""Offline tests for scripts/reproduce_data_check.py (no network, tiny synthetic CSVs)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reproduce_data_check.py"

HEADER = "Date,Capital Gains,Close,Dividends,High,Low,Open,Stock Splits,Volume\n"
ROW1 = "2020-01-02,0.0,10.0,0.0,11.0,9.0,10.0,0.0,100\n"
ROW2 = "2020-01-03,0.0,10.5,0.2,11.0,9.0,10.0,0.0,120\n"
GOOD = HEADER + ROW1 + ROW2
DRIFTED = HEADER + ROW1.replace(",10.0,", ",9.9,", 1) + ROW2.replace(",10.5,", ",10.4,", 1)


@pytest.fixture(scope="module")
def chk():
    spec = importlib.util.spec_from_file_location("reproduce_data_check", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(chk, text: str, tmp_path: Path) -> str:
    p = tmp_path / "_h.csv"
    p.write_bytes(text.encode())
    return chk.sha256_file(p)


def _repo(chk, tmp_path: Path, frozen: dict[str, str]):
    """A fake repo with one frozen manifest whose hashes are those of ``frozen``."""
    root = tmp_path / "repo"
    manifests = root / "data" / "manifests"
    manifests.mkdir(parents=True)
    hashes = {f"{s}.csv": _sha(chk, text, tmp_path) for s, text in frozen.items()}
    manifest = {
        "dataset_id": "fake_v1",
        "source": "yfinance",
        "source_version": "1.7.0",
        "symbols": list(frozen),
        "interval": "1d",
        "requested_start": "2020-01-01",
        "requested_end_exclusive": "2020-02-01",
        "actual_start": {s: "2020-01-02" for s in frozen},
        "actual_end": {s: "2020-01-03" for s in frozen},
        "row_counts": {s: 2 for s in frozen},
        "missing_counts": {s: 0 for s in frozen},
        "actions": {
            s: {
                "dividends": {"present": True, "nonzero_count": 1},
                "stock_splits": {"present": True, "nonzero_count": 0},
                "capital_gains": {"present": True, "nonzero_count": 0},
            }
            for s in frozen
        },
        "retrieval_timestamp_utc": "2026-01-01T00:00:00Z",
        "sha256": {**hashes, "dataset_canonical": chk.canonical_dataset_hash(hashes)},
        "parameters": {"start": "2020-01-01", "end": "2020-02-01", "interval": "1d"},
    }
    path = manifests / "fake_v1.json"
    path.write_text(json.dumps(manifest, indent=2))
    return root, path


def _fetcher(files: dict[str, str]):
    def fetch(spec, out_dir: Path):
        for sym, text in files.items():
            (out_dir / f"{sym}.csv").write_bytes(text.encode())
        return {"retrieval_timestamp_utc": "2026-11-17T00:00:00Z", "excluded": []}

    return fetch


CONFIGS = ({"manifest": "fake_v1.json", "kind": "yfinance_script", "script": "x.py"},)


def _run(chk, root: Path, fetch, tmp_path: Path, extra=()):
    json_out = tmp_path / "out" / "report.json"
    json_out.parent.mkdir(exist_ok=True)
    md_out = tmp_path / "out" / "report.md"
    code = chk.main(
        ["--json-out", str(json_out), "--md-out", str(md_out), *extra],
        fetch=fetch,
        repo_root=root,
        configs=CONFIGS,
    )
    return code, json.loads(json_out.read_text()), md_out.read_text()


def test_match(chk, tmp_path):
    root, _ = _repo(chk, tmp_path, {"AAA": GOOD, "BBB": GOOD})
    code, report, md = _run(chk, root, _fetcher({"AAA": GOOD, "BBB": GOOD}), tmp_path)
    ds = report["datasets"][0]
    assert code == 0
    assert ds["classification"] == "MATCH"
    assert ds["files_matched"] == 2
    assert all(f["match"] is True for f in ds["files"])
    assert ds["dataset_canonical"]["match"] is True
    assert "| fake_v1 |" in md and "MATCH" in md


def test_drift_is_described_and_exit_zero(chk, tmp_path):
    root, _ = _repo(chk, tmp_path, {"AAA": GOOD, "BBB": GOOD})
    code, report, md = _run(chk, root, _fetcher({"AAA": GOOD, "BBB": DRIFTED}), tmp_path)
    ds = report["datasets"][0]
    assert code == 0
    assert ds["classification"] == "DATA DRIFT (not a reproduction failure of the analysis)"
    assert [f["match"] for f in ds["files"]] == [True, False]
    assert ds["dataset_canonical"]["match"] is False
    assert ds["files"][1]["rows"]["match"] is True  # metadata unchanged, values drifted
    assert any("bytes differ in values only" in s for s in ds["drift_summary"])
    assert "DATA DRIFT" in md


def test_metadata_drift_is_reported(chk, tmp_path):
    root, _ = _repo(chk, tmp_path, {"AAA": GOOD})
    longer = GOOD + "2020-01-06,0.0,10.6,0.0,11.0,9.0,10.0,0.0,90\n"
    _, report, _ = _run(chk, root, _fetcher({"AAA": longer}), tmp_path)
    f = report["datasets"][0]["files"][0]
    assert f["rows"] == {"expected": 2, "fresh": 3, "match": False}
    assert f["last_date"]["fresh"] == "2020-01-06"


def test_missing_values_counted(chk, tmp_path):
    text = HEADER + ROW1.replace("0.0,10.0,", "0.0,,", 1) + ROW2
    p = tmp_path / "m.csv"
    p.write_text(text)
    s = chk.summarize_csv(p)
    assert s["missing_values"] == 1
    assert s["actions"]["dividends"]["nonzero_count"] == 1


def test_missing_fresh_file_is_a_drift_not_a_crash(chk, tmp_path):
    root, _ = _repo(chk, tmp_path, {"AAA": GOOD, "BBB": GOOD})
    code, report, _ = _run(chk, root, _fetcher({"AAA": GOOD}), tmp_path)
    ds = report["datasets"][0]
    assert code == 0
    assert ds["files"][1]["fresh_file_present"] is False
    assert ds["files"][1]["match"] is False
    assert ds["classification"].startswith("DATA DRIFT")


def test_fetch_failure_exits_two(chk, tmp_path):
    root, _ = _repo(chk, tmp_path, {"AAA": GOOD})

    def boom(spec, out_dir):
        raise chk.FetchError("network down")

    code, report, md = _run(chk, root, boom, tmp_path)
    assert code == 2
    assert report["datasets"][0]["classification"] == "FETCH FAILED"
    assert "network down" in report["datasets"][0]["error"]
    assert report["exit_code"] == 2 and "FETCH FAILED" in md


def test_refuses_to_write_report_into_manifest_dir(chk, tmp_path):
    root, path = _repo(chk, tmp_path, {"AAA": GOOD})
    before = path.read_text()
    code = chk.main(
        ["--json-out", str(path)], fetch=_fetcher({"AAA": GOOD}), repo_root=root, configs=CONFIGS
    )
    assert code == 2
    assert path.read_text() == before


def test_frozen_manifest_untouched_after_run(chk, tmp_path):
    root, path = _repo(chk, tmp_path, {"AAA": GOOD})
    before = path.read_bytes()
    _run(chk, root, _fetcher({"AAA": DRIFTED}), tmp_path)
    assert path.read_bytes() == before


def test_fetch_script_removes_throwaway_manifest_and_flags_changes(chk, tmp_path, monkeypatch):
    root, path = _repo(chk, tmp_path, {"AAA": GOOD})
    (root / "scripts").mkdir()
    (root / "scripts" / "x.py").write_text("")
    spec = {
        "dataset_id": "fake_v1",
        "script": "x.py",
        "symbols": ["AAA"],
        "start": "2020-01-01",
        "end": "2020-02-01",
        "interval": "1d",
        "extra_args": [],
    }
    throwaway = root / "data" / "manifests" / "fake_v1_reprocheck.json"
    out = tmp_path / "raw"
    out.mkdir()

    def fake_run(cmd, **kwargs):
        assert cmd[cmd.index("--dataset-id") + 1] == "fake_v1_reprocheck"
        assert "--no-freeze" in cmd and "--raw-dir" in cmd
        throwaway.write_text("{}")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(chk, "_run", fake_run)
    chk.fetch_yfinance_script(spec, out, root)
    assert not throwaway.exists()
    assert path.exists()

    def tamper(cmd, **kwargs):
        path.write_text("{}")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(chk, "_run", tamper)
    with pytest.raises(chk.FetchError, match="changed"):
        chk.fetch_yfinance_script(spec, out, root)


def test_fetch_script_refuses_raw_dir_inside_repo(chk, tmp_path):
    root, _ = _repo(chk, tmp_path, {"AAA": GOOD})
    spec = {"dataset_id": "fake_v1", "script": "x.py", "symbols": ["AAA"]}
    with pytest.raises(chk.FetchError, match="inside the repository"):
        chk.fetch_yfinance_script(spec, root / "data" / "raw", root)


def test_canonical_only_manifest_and_excluded_symbols(chk, tmp_path):
    """Sector-equities layout: uniform counts, no per-file hashes, an excluded symbol."""
    root, path = _repo(chk, tmp_path, {"AAA": GOOD, "BBB": GOOD})
    manifest = json.loads(path.read_text())
    canonical = manifest["sha256"]["dataset_canonical"]
    for key in ("row_counts", "actual_start", "actual_end", "missing_counts", "actions"):
        manifest.pop(key)
    manifest.update(
        row_count_uniform=2,
        actual_start_uniform="2020-01-02",
        actual_end_uniform="2020-01-03",
        missing_ohlcv_cells_uniform=0,
        sha256={"dataset_canonical": canonical},
        requested_universe=["AAA", "BBB", "CCC"],
        excluded_at_freeze=[{"symbol": "CCC", "reason": "coverage_gate"}],
    )
    path.write_text(json.dumps(manifest))
    excluded = [{"symbol": "CCC", "reason": "coverage_gate"}]

    def fetch(spec, out_dir):
        assert spec["symbols"] == ["AAA", "BBB", "CCC"]  # requested universe, not survivors
        for sym in ("AAA", "BBB"):
            (out_dir / f"{sym}.csv").write_bytes(GOOD.encode())
        return {"retrieval_timestamp_utc": "2026-11-17T00:00:00Z", "excluded": excluded}

    code, report, _ = _run(chk, root, fetch, tmp_path)
    ds = report["datasets"][0]
    assert code == 0
    assert ds["classification"] == "MATCH"
    assert [f["match"] for f in ds["files"]] == [None, None]  # no per-file hash recorded
    assert ds["dataset_canonical"]["match"] is True
    assert ds["excluded_at_freeze"]["match"] is True

    def drifted(spec, out_dir):
        meta = fetch(spec, out_dir)
        (out_dir / "BBB.csv").write_bytes(DRIFTED.encode())
        return meta

    _, report, _ = _run(chk, root, drifted, tmp_path)
    assert report["datasets"][0]["classification"].startswith("DATA DRIFT")
