"""Low-storage remote access helpers (spec section 24).

- HTTPRangeFile: file-like object over HTTP Range requests (no full download)
- open_remote_zip: zipfile over HTTP ranges (extract single members cheaply)
- stream_tar_zst: stream-decompress a remote .tar.zst shard sequentially,
  yielding tar members WITHOUT storing the shard locally (ProCUA: 853 GB in 50
  shards; shards are ~18.5 GB each and must never be mirrored locally)
- hf_url / hf_tree / fetch_bytes: small Hugging Face helpers with explicit,
  controlled caching (never accumulates silently)
"""
from __future__ import annotations

import io
import os
import tarfile
import zipfile
from typing import BinaryIO, Dict, Iterator, List, Optional

import requests

DEFAULT_CHUNK = 1 << 20  # 1 MiB
HF = "https://huggingface.co"
USER_AGENT = "jxagent-builder/1.0"


def session_with_headers() -> requests.Session:
    s = requests.Session()
    # "Connection: close" avoids stale keep-alive sockets on CDN hosts, which
    # were observed to stall range reads indefinitely on Windows clients.
    s.headers.update({"User-Agent": USER_AGENT, "Connection": "close"})
    return s



REQUEST_TIMEOUT = 30
REQUEST_ATTEMPTS = 3
RETRY_AFTER_CAP = 120  # never sleep longer than this even if asked


class RangeIntegrityError(Exception):
    """A ranged GET returned wrong-offset, short, or full-body content."""


def _retry_delay(resp, attempt: int) -> float:
    """Backoff for a retryable response; honors Retry-After when present."""
    retry_after = resp.headers.get("Retry-After") if resp is not None else None
    if retry_after:
        try:
            return min(max(float(retry_after), 0.0), RETRY_AFTER_CAP)
        except ValueError:
            pass
    return 1.0 * (attempt + 1)


def _is_retryable_status(code: int) -> bool:
    return code in (408, 429, 500, 502, 503, 504)


def get_with_retry(session, url, *, headers=None, timeout=REQUEST_TIMEOUT,
                   attempts=REQUEST_ATTEMPTS, stream=False):
    """GET with bounded retries. CDN connections occasionally stall forever;
    short timeouts + retries turn stalls into recoverable hiccups. 429/5xx
    honor Retry-After when the server provides it."""
    import time
    last = None
    for attempt in range(attempts):
        try:
            r = session.get(url, headers=headers, timeout=timeout, stream=stream)
            if _is_retryable_status(r.status_code) and attempt < attempts - 1:
                delay = _retry_delay(r, attempt)
                r.close()
                time.sleep(delay)
                continue
            return r
        except requests.RequestException as e:
            last = e
            time.sleep(1.0 * (attempt + 1))
    raise last


def verify_ranged_response(resp, start: int, end: int) -> None:
    """Raise RangeIntegrityError unless resp is exactly bytes [start, end].

    Guards against: wrong-offset Content-Range, short 206 bodies, and
    unexpected 200 full-body answers to a Range request (a proxy stripping
    the header). Wrong content must never silently enter the dataset."""
    if resp.status_code == 200:
        raise RangeIntegrityError(
            f"expected 206 for bytes={start}-{end}, got full-body 200 ({resp.url})")
    if resp.status_code != 206:
        # non-206 is reported by raise_for_status() at the caller
        return
    content_range = resp.headers.get("Content-Range", "")
    # form: "bytes <start>-<end>/<total>"
    try:
        range_part = content_range.split(" ", 1)[1].split("/", 1)[0]
        got_start, got_end = (int(x) for x in range_part.split("-", 1))
    except (IndexError, ValueError):
        got_start, got_end = -1, -1
    if got_start != start or got_end != end:
        raise RangeIntegrityError(
            f"Content-Range {content_range!r} != requested bytes={start}-{end} ({resp.url})")
    expected_len = end - start + 1
    body = resp.content
    if len(body) != expected_len:
        raise RangeIntegrityError(
            f"short/long ranged body: {len(body)} bytes, expected {expected_len} "
            f"for bytes={start}-{end} ({resp.url})")


class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable file-like view of a remote URL via Range requests.

    Reads use a read-ahead window (default 256 KiB) so consumers that read in
    small chunks (e.g. zipfile) do not trigger one HTTP request per chunk —
    that turns a few-MB member extraction into thousands of round trips.
    """

    READAHEAD = 4 << 20  # large aligned reads: small ranged GETs stall on some CDN edges

    def __init__(self, url: str, session: Optional[requests.Session] = None):
        self.url = url
        self.session = session or session_with_headers()
        self.pos = 0
        self._buf = b""
        self._buf_start = 0
        r = self.session.head(url, timeout=60, allow_redirects=True)
        r.raise_for_status()
        self.size = int(r.headers.get("Content-Length", 0))
        if self.size <= 0:
            raise ValueError(f"no Content-Length for {url}")
        if not self._accepts_ranges(r):
            raise ValueError(f"server does not accept ranges for {url}")

    @staticmethod
    def _accepts_ranges(resp) -> bool:
        return (resp.headers.get("Accept-Ranges", "").lower() == "bytes"
                or "bytes" in resp.headers.get("Accept-Ranges", "").lower())

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        if not (self._buf_start <= self.pos <= self._buf_start + len(self._buf)):
            self._buf = b""  # dropped window
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0 or self.pos >= self.size:
            return b""
        if not (self._buf_start <= self.pos < self._buf_start + len(self._buf)):
            fetch = max(n, self.READAHEAD)
            end = min(self.pos + fetch, self.size) - 1
            r = get_with_retry(self.session, self.url,
                               headers={"Range": f"bytes={self.pos}-{end}"})
            r.raise_for_status()
            verify_ranged_response(r, self.pos, end)
            self._buf = r.content
            self._buf_start = self.pos
        offset = self.pos - self._buf_start
        out = self._buf[offset:offset + n]
        self.pos += len(out)
        return out


def open_remote_zip(url: str, session: Optional[requests.Session] = None) -> zipfile.ZipFile:
    return zipfile.ZipFile(HTTPRangeFile(url, session))


def hf_url(repo_id: str, filename: str, repo_type: str = "dataset", revision: str = "main") -> str:
    prefix = "datasets/" if repo_type == "dataset" else ""
    return f"{HF}/{prefix}{repo_id}/resolve/{revision}/{filename}"


def hf_tree(repo_id: str, path: str = "", repo_type: str = "dataset", revision: str = "main",
            session: Optional[requests.Session] = None) -> List[Dict]:
    """List a directory of a HF repo (one level)."""
    session = session or session_with_headers()
    prefix = "datasets/" if repo_type == "dataset" else ""
    url = f"{HF}/api/{prefix}{repo_id}/tree/{revision}/{path}".rstrip("/")
    r = session.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def hf_tree_recursive(repo_id: str, path: str = "", repo_type: str = "dataset",
                      session: Optional[requests.Session] = None,
                      max_entries: int = 200000) -> List[Dict]:
    """Recursively walk a HF repo tree via the API."""
    session = session or session_with_headers()
    out: List[Dict] = []
    stack = [path]
    while stack and len(out) < max_entries:
        cur = stack.pop()
        for item in hf_tree(repo_id, cur, repo_type=repo_type, session=session):
            if item.get("type") == "directory":
                stack.append(item["path"])
            else:
                out.append(item)
    return out


def fetch_bytes(url: str, *, max_bytes: int = 64 << 20,
                session: Optional[requests.Session] = None) -> bytes:
    session = session or session_with_headers()
    r = get_with_retry(session, url, stream=True)
    r.raise_for_status()
    buf = io.BytesIO()
    for chunk in r.iter_content(DEFAULT_CHUNK):
        buf.write(chunk)
        if buf.tell() > max_bytes:
            raise ValueError(f"response exceeds max_bytes ({url})")
    return buf.getvalue()


class ZstHttpResponseReader:
    """Sequential reader over an HTTP byte stream of a .tar.zst archive.

    Uses a streaming zstd decompressobj (NOT one-shot decompress): tar.zst
    shards use frames without an embedded content size, which one-shot
    decompression rejects ("could not determine content size in frame
    header")."""

    def __init__(self, chunk_iter):
        import zstandard
        self.chunks = iter(chunk_iter)
        self._dobj = zstandard.ZstdDecompressor().decompressobj()
        self._buf = b""
        self._done = False

    def read(self, size=-1):
        if size is None or size < 0:
            size = 1 << 24
        while len(self._buf) < size and not self._done:
            try:
                chunk = next(self.chunks)
            except StopIteration:
                self._done = True
                self._buf += self._dobj.flush()
                break
            if not chunk:
                continue
            self._buf += self._dobj.decompress(chunk)
        out, self._buf = self._buf[:size], self._buf[size:]
        return out


class RangedChunkSource:
    """Sequential compressed bytes via bounded ranged GETs.

    A single long-lived streaming GET can stall forever mid-body (some CDN
    edges trickle bytes, defeating read timeouts). Fixed-size range requests
    are individually retryable from their exact offset, which makes large
    shard streaming reliable."""

    def __init__(self, url, session=None, chunk_size=8 << 20):
        self.url = url
        self.session = session or session_with_headers()
        self.chunk_size = chunk_size
        self.offset = 0
        self.size = None
        r = self.session.head(self.url, timeout=60, allow_redirects=True)
        r.raise_for_status()
        self.size = int(r.headers.get("Content-Length", 0))
        if self.size <= 0:
            raise ValueError(f"no Content-Length for {url}")

    def __iter__(self):
        while self.offset < self.size:
            end = min(self.offset + self.chunk_size, self.size) - 1
            expected = end - self.offset + 1
            last = None
            data = b""
            for attempt in range(REQUEST_ATTEMPTS):
                try:
                    r = self.session.get(self.url, timeout=REQUEST_TIMEOUT,
                                         headers={"Range": f"bytes={self.offset}-{end}"})
                    if _is_retryable_status(r.status_code) and attempt < REQUEST_ATTEMPTS - 1:
                        import time as _t
                        delay = _retry_delay(r, attempt)
                        r.close()
                        _t.sleep(delay)
                        continue
                    r.raise_for_status()
                    verify_ranged_response(r, self.offset, end)
                    data = r.content
                    if len(data) != expected:
                        raise RangeIntegrityError(
                            f"ranged chunk length {len(data)} != {expected}")
                    self.offset += len(data)
                    yield data
                    last = None
                    break
                except (requests.RequestException, RangeIntegrityError) as e:
                    # Integrity mismatches can be transient CDN/proxy failures,
                    # but they must never advance the offset. Retry the exact
                    # same byte range and fail closed after the bounded budget.
                    last = e
                    try:
                        r.close()
                    except Exception:
                        pass
                    import time as _t
                    _t.sleep(1.0 * (attempt + 1))
            if last is not None:
                raise last
            if not data:
                return


def stream_tar_zst(url: str, max_members: Optional[int] = None,
                   session: Optional[requests.Session] = None,
                   byte_limit: Optional[int] = None) -> Iterator[tarfile.TarInfo]:
    """Yield members of a remote tar.zst WITHOUT storing the archive.

    Uses sequential 'r|' tar mode over an HTTP stream + zstd decompressor.
    Stop consuming the generator at any time (smoke tests) — only the bytes
    actually read are transferred.
    """
    source = RangedChunkSource(url, session=session)
    reader = ZstHttpResponseReader(iter(source))
    tf = tarfile.open(fileobj=reader, mode="r|")
    n = 0
    read_bytes = 0
    for member in tf:
        yield member, tf
        n += 1
        read_bytes += max(member.size, 0)
        if max_members is not None and n >= max_members:
            break
        if byte_limit is not None and read_bytes > byte_limit:
            break
    tf.close()


def download_file(url: str, dest: str, session: Optional[requests.Session] = None,
                  max_bytes: Optional[int] = None) -> str:
    """Controlled download to an explicit destination (no hidden HF cache).

    Retried on transport errors/429/5xx; the written size is verified against
    Content-Length when the server provides it, so a truncated or corrupted
    download can never silently become the destination file."""
    import time
    session = session or session_with_headers()
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    last = None
    for attempt in range(REQUEST_ATTEMPTS):
        try:
            r = get_with_retry(session, url, stream=True, timeout=600)
            if _is_retryable_status(r.status_code) and attempt < REQUEST_ATTEMPTS - 1:
                delay = _retry_delay(r, attempt)
                r.close()
                time.sleep(delay)
                continue
            r.raise_for_status()
            expected = r.headers.get("Content-Length")
            expected = int(expected) if expected and expected.isdigit() else None
            total = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    total += len(chunk)
                    if max_bytes is not None and total > max_bytes:
                        f.close()
                        os.remove(tmp)
                        raise ValueError(f"download exceeds max_bytes ({url})")
            if expected is not None and total != expected:
                os.remove(tmp)
                raise RangeIntegrityError(
                    f"downloaded {total} bytes, Content-Length said {expected} ({url})")
            os.replace(tmp, dest)
            return dest
        except (requests.RequestException, RangeIntegrityError) as e:
            last = e
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            time.sleep(1.0 * (attempt + 1))
    raise last
