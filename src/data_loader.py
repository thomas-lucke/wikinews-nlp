"""Download or clone the raw Wikinews dataset onto disk."""

import logging
import shutil
import subprocess
import tarfile
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

DATA_EXTENSIONS = {".json", ".jsonl", ".csv", ".tsv", ".txt"}
SKIP_SIZE_THRESHOLD_BYTES = 100 * 1024
DIRECT_FILE_EXTENSIONS = (".zip", ".tar.gz", ".tgz", ".gz", ".csv", ".json", ".jsonl")
REQUEST_TIMEOUT = 60
RETRY_DELAY_SECONDS = 5


def _has_sufficient_data(raw_path: Path) -> bool:
    if not raw_path.exists() or not raw_path.is_dir():
        return False
    data_files = [
        p for p in raw_path.rglob("*") if p.is_file() and p.suffix.lower() in DATA_EXTENSIONS
    ]
    if not data_files:
        return False
    total_size = sum(p.stat().st_size for p in data_files)
    return total_size > SKIP_SIZE_THRESHOLD_BYTES


def _strip_single_top_dir(target_dir: Path) -> None:
    entries = [p for p in target_dir.iterdir() if p.name not in {".", ".."}]
    if len(entries) == 1 and entries[0].is_dir():
        top = entries[0]
        for child in list(top.iterdir()):
            shutil.move(str(child), str(target_dir / child.name))
        top.rmdir()


def _extract_archive(archive_path: Path, target_dir: Path) -> None:
    name = archive_path.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(target_dir)
    elif name.endswith(".tar.gz") or name.endswith(".tgz"):
        with tarfile.open(archive_path, mode="r:gz") as tf:
            tf.extractall(target_dir)
    elif name.endswith(".gz"):
        import gzip

        out_path = target_dir / archive_path.stem
        with gzip.open(archive_path, "rb") as src, out_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return
    else:
        raise RuntimeError(f"Unsupported archive type: {archive_path.name}")
    _strip_single_top_dir(target_dir)


def _stream_download(url: str, dest: Path) -> requests.Response:
    """Download url to dest with one retry on Timeout/ConnectionError.

    Raises RuntimeError on HTTP 4xx (no retry), or after the retry is also exhausted.
    """
    attempts = 0
    last_exc: Exception | None = None
    while attempts < 2:
        attempts += 1
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            logger.warning(
                "Network error on attempt %d for %s: %s", attempts, url, exc
            )
            if attempts < 2:
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            raise RuntimeError(
                f"Failed to download {url} after retry: {exc}"
            ) from exc

        if 400 <= response.status_code < 500:
            response.close()
            raise RuntimeError(
                f"HTTP {response.status_code} for {url} (no retry on 4xx)"
            )
        if response.status_code != 200:
            response.close()
            raise RuntimeError(
                f"HTTP {response.status_code} for {url}"
            )

        with dest.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if chunk:
                    fh.write(chunk)
        response.close()
        return response

    raise RuntimeError(f"Failed to download {url}: {last_exc}")


def _git_available() -> bool:
    return shutil.which("git") is not None


def _try_git_clone(source_url: str, raw_path: Path) -> bool:
    if raw_path.exists() and any(raw_path.iterdir()):
        for child in list(raw_path.iterdir()):
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", source_url, str(raw_path)],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False
    if result.returncode != 0:
        logger.warning("git clone failed: %s", result.stderr.strip())
        return False
    logger.info("Cloned %s into %s", source_url, raw_path)
    return True


def _github_zip_fallback(source_url: str, raw_path: Path) -> None:
    parsed = urlparse(source_url.rstrip("/"))
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise RuntimeError(
            f"Cannot derive owner/repo from GitHub URL: {source_url}"
        )
    owner, repo = parts[0], parts[1].removesuffix(".git")

    last_error: Exception | None = None
    for branch in ("main", "master"):
        zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
        archive_path = raw_path / f"_download_{branch}.zip"
        try:
            _stream_download(zip_url, archive_path)
        except RuntimeError as exc:
            last_error = exc
            logger.warning("ZIP fallback failed for branch %s: %s", branch, exc)
            if archive_path.exists():
                archive_path.unlink()
            continue
        try:
            _extract_archive(archive_path, raw_path)
        finally:
            if archive_path.exists():
                archive_path.unlink()
        logger.info("Downloaded ZIP for %s/%s@%s", owner, repo, branch)
        return

    raise RuntimeError(
        f"GitHub ZIP fallback exhausted for {source_url}. "
        f"Last error: {last_error}. Set source_url to a direct archive URL."
    )


def download_dataset(source_url: str, raw_path: str) -> Path:
    """Download or clone source_url into raw_path. Returns absolute Path(raw_path)."""
    target = Path(raw_path)
    target.mkdir(parents=True, exist_ok=True)
    resolved = target.resolve()

    if _has_sufficient_data(target):
        logger.info(
            "Skipping download: %s already contains data files > %d bytes total.",
            resolved,
            SKIP_SIZE_THRESHOLD_BYTES,
        )
        return resolved

    if "github.com" in source_url:
        used_git = False
        if _git_available():
            used_git = _try_git_clone(source_url, target)
        if not used_git:
            try:
                _github_zip_fallback(source_url, target)
            except RuntimeError as exc:
                if not _git_available():
                    raise EnvironmentError(
                        f"git is unavailable and ZIP fallback failed: {exc}"
                    ) from exc
                raise
        return resolved

    lower_url = source_url.lower()
    if lower_url.endswith(DIRECT_FILE_EXTENSIONS):
        filename = Path(urlparse(source_url).path).name or "download.bin"
        dest = target / filename
        _stream_download(source_url, dest)
        if filename.lower().endswith((".zip", ".tar.gz", ".tgz", ".gz")):
            try:
                _extract_archive(dest, target)
            except (zipfile.BadZipFile, tarfile.TarError, OSError) as exc:
                raise RuntimeError(f"Extraction failed for {dest}: {exc}") from exc
            finally:
                if dest.exists() and dest.suffix.lower() in {".zip", ".gz", ".tgz"}:
                    dest.unlink()
        return resolved

    filename = Path(urlparse(source_url).path).name or "download.bin"
    dest = target / filename
    _stream_download(source_url, dest)
    return resolved
