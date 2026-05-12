"""Tests for src/data_loader.py.

SPEC: data_loader has a single responsibility — get raw files onto disk.
Tests cover the skip heuristic, network retry behaviour, and the GitHub
ZIP fallback. All network and subprocess interactions are mocked.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

data_loader = pytest.importorskip("src.data_loader")


# ---------------------------------------------------------------------------
# Skip-condition heuristic
# ---------------------------------------------------------------------------

def test_skips_download_when_raw_path_has_large_recognised_data_file(tmp_path):
    # SPEC: download_dataset — "Skip condition: raw_path already exists AND contains
    # at least 1 file with a recognised data extension (.json, .jsonl, .csv, .tsv, .txt)
    # AND total size of recognised data files is > 100KB."
    raw = tmp_path / "raw"
    raw.mkdir()
    big_jsonl = raw / "wikinews.jsonl"
    big_jsonl.write_bytes(b'{"text": "x"}\n' * 20000)  # > 100KB
    assert big_jsonl.stat().st_size > 100 * 1024

    with patch.object(data_loader, "requests", create=True) as mock_requests, \
         patch.object(data_loader, "subprocess", create=True) as mock_subproc:
        result = data_loader.download_dataset("https://example.com/x.zip", str(raw))
        mock_requests.get.assert_not_called()
        mock_subproc.run.assert_not_called()
    assert isinstance(result, Path)
    assert result.resolve() == raw.resolve()


def test_does_not_skip_when_directory_is_empty(tmp_path):
    # SPEC: download_dataset — "Both conditions must be true." An empty directory
    # (only .gitkeep would satisfy neither) must not trigger the skip path.
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / ".gitkeep").write_text("", encoding="utf-8")

    with patch.object(data_loader, "requests", create=True) as mock_requests:
        mock_response = MagicMock(status_code=200, iter_content=lambda chunk_size: [b""])
        mock_response.raise_for_status = MagicMock()
        mock_requests.get.return_value.__enter__.return_value = mock_response
        mock_requests.get.return_value = mock_response
        try:
            data_loader.download_dataset("https://example.com/data.jsonl", str(raw))
        except Exception:
            # We do not care whether download succeeds — only that the skip
            # heuristic did NOT short-circuit and the network was contacted.
            pass
        assert mock_requests.get.called, (
            "Empty raw_path must trigger a download attempt, not skip."
        )


def test_skip_check_ignores_non_data_files(tmp_path):
    # SPEC: download_dataset — recognised data extensions are
    # ".json, .jsonl, .csv, .tsv, .txt". README.md alone must not count.
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "README.md").write_bytes(b"x" * (200 * 1024))  # > 100KB but wrong ext

    with patch.object(data_loader, "requests", create=True) as mock_requests:
        mock_requests.get.side_effect = Exception("forced — confirms download attempted")
        with pytest.raises(Exception):
            data_loader.download_dataset("https://example.com/data.jsonl", str(raw))
        assert mock_requests.get.called


# ---------------------------------------------------------------------------
# Network behaviour
# ---------------------------------------------------------------------------

def test_returns_runtime_error_on_repeated_network_failure(tmp_path):
    # SPEC: download_dataset — "Retry: one retry on network failure ... Raises:
    # RuntimeError: If download fails after one retry".
    raw = tmp_path / "raw"
    raw.mkdir()
    import requests as real_requests  # used only to access exception classes

    with patch.object(data_loader, "requests", create=True) as mock_requests:
        mock_requests.ConnectionError = real_requests.ConnectionError
        mock_requests.Timeout = real_requests.Timeout
        mock_requests.get.side_effect = real_requests.ConnectionError("boom")
        with pytest.raises(RuntimeError):
            data_loader.download_dataset(
                "https://example.com/data.jsonl", str(raw)
            )


def test_returns_path_object_not_string(tmp_path):
    # SPEC: download_dataset — "Returns: Absolute Path to raw_path."
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "data.jsonl").write_bytes(b"x" * (200 * 1024))

    result = data_loader.download_dataset("https://example.com/x.jsonl", str(raw))
    assert isinstance(result, Path), "Must return pathlib.Path, not str"


def test_http_4xx_does_not_trigger_retry(tmp_path):
    # SPEC: download_dataset — "Failure cases that do NOT retry: HTTP 4xx
    # (client error), extraction failure."
    raw = tmp_path / "raw"
    raw.mkdir()
    import requests as real_requests

    call_counter = {"n": 0}

    def fake_get(*args, **kwargs):
        call_counter["n"] += 1
        response = MagicMock()
        response.status_code = 404
        response.raise_for_status.side_effect = real_requests.HTTPError("404")
        return response

    with patch.object(data_loader, "requests", create=True) as mock_requests:
        mock_requests.ConnectionError = real_requests.ConnectionError
        mock_requests.Timeout = real_requests.Timeout
        mock_requests.HTTPError = real_requests.HTTPError
        mock_requests.get.side_effect = fake_get
        with pytest.raises((RuntimeError, real_requests.HTTPError)):
            data_loader.download_dataset(
                "https://example.com/data.jsonl", str(raw)
            )
        assert call_counter["n"] == 1, (
            "HTTP 4xx must not be retried (spec excludes 4xx from retry policy)."
        )


def test_github_zip_fallback_raises_when_neither_branch_works(tmp_path):
    # SPEC: download_dataset — "The ZIP URL is constructed by trying these
    # branch names in order: main, master. If neither returns HTTP 200, raise
    # RuntimeError ..."
    raw = tmp_path / "raw"
    raw.mkdir()
    import requests as real_requests

    def fake_get(url, *args, **kwargs):
        response = MagicMock()
        response.status_code = 404
        response.raise_for_status.side_effect = real_requests.HTTPError("404")
        return response

    with patch.object(data_loader, "requests", create=True) as mock_requests, \
         patch.object(data_loader, "subprocess", create=True) as mock_subproc:
        mock_requests.ConnectionError = real_requests.ConnectionError
        mock_requests.Timeout = real_requests.Timeout
        mock_requests.HTTPError = real_requests.HTTPError
        mock_requests.get.side_effect = fake_get
        # Simulate git not present so the ZIP fallback path is exercised.
        mock_subproc.run.side_effect = FileNotFoundError("git not on PATH")
        with pytest.raises((RuntimeError, EnvironmentError)):
            data_loader.download_dataset(
                "https://github.com/example/repo", str(raw)
            )
