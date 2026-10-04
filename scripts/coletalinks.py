# ============================================================
# GOOGLE AI OVERVIEW — CITED SOURCE COLLECTOR
#
# Features:
# - Parallel collection with ThreadPoolExecutor
# - Resume from previous runs
# - Same source_id scheme as previous script
# - Hard wall-clock timeout using curl subprocess
# - Per-host rate limiting
# - robots.txt support + cache
# - Retries
# - Atomic metadata writes
# - HTML / PDF / text extraction
# - Deduplicated URL corpus
# - Saves raw files + extracted text + metadata
# - Can safely be interrupted and restarted
#
# ============================================================

from pathlib import Path
from urllib.parse import (
    urlparse,
    urlunparse,
    parse_qsl,
    urlencode,
)
from urllib import robotparser

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

from datetime import datetime, timezone

import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import traceback

import pandas as pd

import tldextract
import trafilatura

from bs4 import BeautifulSoup

import pymupdf


# ============================================================
# 0. CONFIG
# ============================================================

BASE_DIR = Path(
    "/scratch/victoria.estanislau/ai-summary"
)

COLLECTION_VERSION = "v1_dallas"


AIO_COLLECTION_DIR = (
    BASE_DIR
    / "annotations"
    / COLLECTION_VERSION
    / "google_aio_collection"
)


# ------------------------------------------------------------
# Collected source corpus = DATA
# ------------------------------------------------------------

CORPUS_DIR = (
    BASE_DIR
    / "annotations"
    / COLLECTION_VERSION
    / "source_corpus"
)

RAW_DIR = (
    CORPUS_DIR
    / "raw"
)

TEXT_DIR = (
    CORPUS_DIR
    / "text"
)

METADATA_DIR = (
    CORPUS_DIR
    / "metadata"
)


# ------------------------------------------------------------
# Collection diagnostics = RESULTS
# ------------------------------------------------------------

RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "source_collection_dallas"
)


for directory in [
    CORPUS_DIR,
    RAW_DIR,
    TEXT_DIR,
    METADATA_DIR,
    RESULTS_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# 1. CRAWLER CONFIG
# ============================================================

# Total concurrent URLs.
MAX_WORKERS = 8


# Minimum interval between request STARTS
# to the SAME hostname.
#
# Different hosts are still accessed concurrently.
DELAY_PER_HOST_SECONDS = 0.5


# True wall-clock limit for each PAGE download.
HARD_URL_TIMEOUT_SECONDS = 45


# robots.txt gets a smaller timeout.
ROBOTS_HARD_TIMEOUT_SECONDS = 10


# Prevent accidental giant downloads.
MAX_DOWNLOAD_BYTES = (
    25 * 1024 * 1024
)


# robots.txt should be tiny.
MAX_ROBOTS_BYTES = (
    1 * 1024 * 1024
)


# curl retry count for transient failures.
CURL_RETRIES = 1


RESPECT_ROBOTS_TXT = True


USER_AGENT = os.environ.get(
    "AIO_RESEARCH_USER_AGENT",
    (
        "Google-AIO-Audit-Research/1.0 "
        "(academic research)"
    ),
)


# ------------------------------------------------------------
# IMPORTANT RESUME BEHAVIOR
# ------------------------------------------------------------

# False:
#     ANY valid metadata JSON already present counts as done.
#
#     This includes:
#       success
#       403
#       robots_disallowed
#       timeout
#       empty_text
#       etc.
#
#     => exactly what you want now.
#
# True:
#     retry previously failed/incomplete pages.
#
RETRY_FAILED_EXISTING = False


# ============================================================
# 2. URL NORMALIZATION
#
# IMPORTANT:
# Keep this compatible with the previous collector so already
# generated source_id values remain identical.
# ============================================================

TRACKING_PARAMETERS = {
    "gclid",
    "fbclid",
    "msclkid",
    "mc_cid",
    "mc_eid",
    "_ga",
    "_gl",
}


def is_tracking_parameter(key):

    key = (
        key
        .strip()
        .lower()
    )

    return (
        key.startswith("utm_")
        or key in TRACKING_PARAMETERS
    )


def normalize_url(url):

    url = (
        url
        .strip()
        .rstrip(".,;")
    )

    parsed = urlparse(
        url
    )


    scheme = (
        parsed.scheme.lower()
        or "https"
    )


    hostname = (
        parsed.hostname.lower()
        if parsed.hostname
        else ""
    )


    if hostname.startswith("www."):
        hostname = hostname[4:]


    try:

        port = parsed.port

    except ValueError:

        port = None


    if (
        port
        and not (
            scheme == "https"
            and port == 443
        )
        and not (
            scheme == "http"
            and port == 80
        )
    ):

        netloc = (
            f"{hostname}:{port}"
        )

    else:

        netloc = hostname


    path = (
        parsed.path
        or "/"
    )


    path = re.sub(
        r"/+",
        "/",
        path,
    )


    if (
        len(path) > 1
        and path.endswith("/")
    ):

        path = path[:-1]


    query_items = []


    for key, value in parse_qsl(
        parsed.query,
        keep_blank_values=True,
    ):

        if is_tracking_parameter(
            key
        ):
            continue

        query_items.append(
            (
                key,
                value,
            )
        )


    query_items = sorted(
        query_items
    )


    query = urlencode(
        query_items,
        doseq=True,
    )


    return urlunparse(
        (
            scheme,
            netloc,
            path,
            "",
            query,
            "",
        )
    )


# ============================================================
# 3. STABLE SOURCE ID
#
# Same as previous script.
# ============================================================

def source_id_for_url(url):

    return hashlib.sha256(
        url.encode(
            "utf-8"
        )
    ).hexdigest()[:20]


# ============================================================
# 4. TXT FILE PARSING
# ============================================================

def normalize_newlines(text):

    return (
        text
        .replace(
            "\r\n",
            "\n"
        )
        .replace(
            "\r",
            "\n"
        )
    )


def get_section(
    text,
    header,
    next_headers,
):

    text = normalize_newlines(
        text
    )


    match = re.search(
        rf"(?m)^\s*"
        rf"{re.escape(header)}"
        rf"\s*$",
        text,
    )


    if not match:
        return ""


    remainder = text[
        match.end():
    ]


    positions = []


    for next_header in (
        next_headers
    ):

        m = re.search(
            rf"(?m)^\s*"
            rf"{re.escape(next_header)}"
            rf"\s*$",
            remainder,
        )

        if m:
            positions.append(
                m.start()
            )


    if positions:

        remainder = remainder[
            :min(positions)
        ]


    lines = (
        remainder
        .splitlines()
    )


    while (
        lines
        and (
            not lines[0].strip()
            or re.fullmatch(
                r"=+",
                lines[0].strip(),
            )
        )
    ):

        lines.pop(0)


    while (
        lines
        and (
            not lines[-1].strip()
            or re.fullmatch(
                r"=+",
                lines[-1].strip(),
            )
        )
    ):

        lines.pop()


    return "\n".join(
        lines
    ).strip()


def parse_key_values(section):

    result = {}


    for line in (
        section
        or ""
    ).splitlines():

        if ":" not in line:
            continue


        key, value = line.split(
            ":",
            1,
        )


        result[
            key.strip()
        ] = value.strip()


    return result


# ============================================================
# 5. LINKS PARSER
# ============================================================

def parse_links(section):

    links = []

    fallback_index = 1


    for raw_line in (
        section
        or ""
    ).splitlines():

        line = raw_line.strip()


        if not line:
            continue


        source_order = None

        url = None


        # ----------------------------------------------------
        # FORMAT 1:
        #
        # 1 - https://example.com
        # ----------------------------------------------------

        match = re.match(
            r"^(\d+)\s*-\s*(.*)$",
            line,
        )


        if match:

            source_order = int(
                match.group(1)
            )


            rest = (
                match
                .group(2)
                .strip()
            )


            if not rest:
                continue


            markdown = re.search(
                r"\((https?://[^)]+)\)",
                rest,
            )


            if markdown:

                url = (
                    markdown
                    .group(1)
                )

            else:

                normal = re.search(
                    r"https?://\S+",
                    rest,
                )


                if normal:

                    url = (
                        normal
                        .group(0)
                    )


        # ----------------------------------------------------
        # FORMAT 2:
        #
        # [1] [example.com](https://example.com)
        # ----------------------------------------------------

        if url is None:

            match = re.match(
                r"^\[(\d+)\]\s*"
                r"\[[^\]]+\]"
                r"\((https?://[^)]+)\)",
                line,
            )


            if match:

                source_order = int(
                    match.group(1)
                )

                url = (
                    match.group(2)
                )


        # ----------------------------------------------------
        # FORMAT 3:
        #
        # [1] https://example.com
        # ----------------------------------------------------

        if url is None:

            match = re.match(
                r"^\[(\d+)\]\s*"
                r"(https?://\S+)",
                line,
            )


            if match:

                source_order = int(
                    match.group(1)
                )

                url = (
                    match.group(2)
                )


        # ----------------------------------------------------
        # Generic fallback
        # ----------------------------------------------------

        if url is None:

            match = re.search(
                r"https?://\S+",
                line,
            )


            if match:

                source_order = (
                    fallback_index
                )

                url = (
                    match.group(0)
                )


        if url:

            url = url.rstrip(
                ".,;"
            )


            normalized_url = (
                normalize_url(
                    url
                )
            )


            links.append({

                "source_order":
                    source_order,

                "raw_url":
                    url,

                "normalized_url":
                    normalized_url,

                "source_id":
                    source_id_for_url(
                        normalized_url
                    ),
            })


            fallback_index += 1


    return links


# ============================================================
# 6. PARSE ALL GOOGLE AIO FILES
# ============================================================

print()
print("=" * 80)
print("PARSING AIO SOURCE URLS")
print("=" * 80)


usage_rows = []


for filepath in sorted(
    AIO_COLLECTION_DIR.rglob(
        "*.txt"
    )
):

    raw = filepath.read_text(
        encoding="utf-8",
        errors="replace",
    )


    query = get_section(
        raw,
        "QUERY",
        [
            "LINKS",
            "AI OVERVIEW TEXT",
            "COLLECTION INFO",
            "METADATA — DO NOT EDIT",
        ],
    )


    links_section = get_section(
        raw,
        "LINKS",
        [
            "AI OVERVIEW TEXT",
            "COLLECTION INFO",
            "METADATA — DO NOT EDIT",
        ],
    )


    metadata_section = get_section(
        raw,
        "METADATA — DO NOT EDIT",
        [],
    )


    metadata = (
        parse_key_values(
            metadata_section
        )
    )


    links = parse_links(
        links_section
    )


    for link in links:

        usage_rows.append({

            "source_id":
                link[
                    "source_id"
                ],

            "raw_url":
                link[
                    "raw_url"
                ],

            "normalized_url":
                link[
                    "normalized_url"
                ],

            "source_order":
                link[
                    "source_order"
                ],

            "file":
                str(
                    filepath.relative_to(
                        AIO_COLLECTION_DIR
                    )
                ),

            "query":
                query,

            "group":
                metadata.get(
                    "Group",
                    "",
                ),

            "condition":
                metadata.get(
                    "Condition",
                    "",
                ),

            "dimension":
                metadata.get(
                    "Dimension",
                    "",
                ),

            "domain":
                metadata.get(
                    "Domain",
                    "",
                ),

            "outcome":
                metadata.get(
                    "Outcome",
                    "",
                ),

            "query_id":
                metadata.get(
                    "Query ID",
                    "",
                ),
        })


usage = pd.DataFrame(
    usage_rows
)


if usage.empty:

    raise RuntimeError(
        "No source URLs found."
    )


usage.to_csv(
    CORPUS_DIR
    / "url_usage.csv",
    index=False,
)


# ============================================================
# 7. UNIQUE SOURCE LIST
# ============================================================

unique_sources = (
    usage[
        [
            "source_id",
            "normalized_url",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        "source_id"
    )
    .reset_index(
        drop=True
    )
)


print(
    f"\nTotal source occurrences: "
    f"{len(usage):,}"
)


print(
    f"Unique normalized URLs: "
    f"{len(unique_sources):,}"
)


# ============================================================
# 8. REGISTRABLE DOMAIN
# ============================================================

tld_extractor = (
    tldextract.TLDExtract(
        suffix_list_urls=None
    )
)


def registrable_domain(url):

    hostname = (
        urlparse(url)
        .hostname
    )


    if not hostname:
        return None


    hostname = (
        hostname.lower()
    )


    result = (
        tld_extractor(
            hostname
        )
    )


    if (
        result.domain
        and result.suffix
    ):

        return (
            f"{result.domain}."
            f"{result.suffix}"
        )


    return hostname


# ============================================================
# 9. PER-HOST RATE LIMITING
# ============================================================

host_state_guard = (
    threading.Lock()
)

host_request_locks = {}

last_request_by_host = {}


def get_host_request_lock(host):

    with host_state_guard:

        if host not in (
            host_request_locks
        ):

            host_request_locks[
                host
            ] = (
                threading.Lock()
            )


        return (
            host_request_locks[
                host
            ]
        )


def wait_for_host(host):
    """
    Ensure DELAY_PER_HOST_SECONDS between request STARTS
    to the same hostname.

    Different hosts remain fully concurrent.
    """

    if not host:
        return


    lock = (
        get_host_request_lock(
            host
        )
    )


    with lock:

        previous = (
            last_request_by_host
            .get(host)
        )


        if previous is not None:

            elapsed = (
                time.monotonic()
                - previous
            )


            remaining = (
                DELAY_PER_HOST_SECONDS
                - elapsed
            )


            if remaining > 0:

                time.sleep(
                    remaining
                )


        last_request_by_host[
            host
        ] = (
            time.monotonic()
        )


# ============================================================
# 10. HARD-TIMEOUT CURL DOWNLOADER
# ============================================================

def curl_download(
    url,
    hard_timeout,
    max_bytes,
):
    """
    Perform one HTTP download using curl.

    Two independent timeout mechanisms:

    1. curl --max-time
    2. subprocess.run(timeout=...)

    Therefore a pathological remote server cannot leave
    a Python worker blocked indefinitely.
    """

    temp_path = None


    try:

        with tempfile.NamedTemporaryFile(
            delete=False
        ) as temp_file:

            temp_path = (
                temp_file.name
            )


        delimiter = (
            "__AIO_CURL_META__"
        )


        write_out = (
            f"{delimiter}"
            "%{http_code}"
            f"{delimiter}"
            "%{url_effective}"
            f"{delimiter}"
            "%{content_type}"
        )


        command = [

            "curl",

            # Redirects
            "--location",

            # Quiet, but retain error text
            "--silent",
            "--show-error",

            # --------------------------------------------
            # HARD TIMEOUTS
            # --------------------------------------------

            "--max-time",
            str(
                hard_timeout
            ),

            "--connect-timeout",
            "8",

            # --------------------------------------------
            # RETRIES
            # --------------------------------------------

            "--retry",
            str(
                CURL_RETRIES
            ),

            "--retry-delay",
            "1",

            "--retry-max-time",
            str(
                hard_timeout
            ),

            # --------------------------------------------
            # SIZE CONTROL
            # --------------------------------------------

            "--max-filesize",
            str(
                max_bytes
            ),

            # --------------------------------------------
            # HTTP HEADERS
            # --------------------------------------------

            "--user-agent",
            USER_AGENT,

            "--header",
            (
                "Accept: "
                "text/html,"
                "application/xhtml+xml,"
                "application/pdf,"
                "text/plain;q=0.9,"
                "*/*;q=0.5"
            ),

            "--header",
            (
                "Accept-Language: "
                "en-US,en;q=0.9"
            ),

            # --------------------------------------------
            # OUTPUT
            # --------------------------------------------

            "--output",
            temp_path,

            "--write-out",
            write_out,

            url,
        ]


        try:

            completed = subprocess.run(

                command,

                capture_output=True,

                text=True,

                # True Python-side wall-clock cap.
                timeout=(
                    hard_timeout
                    + 10
                ),
            )


        except subprocess.TimeoutExpired:

            return {

                "success":
                    False,

                "status":
                    "hard_timeout",

                "final_url":
                    url,

                "error":
                    (
                        "curl subprocess exceeded "
                        f"{hard_timeout + 10}s"
                    ),
            }


        stdout = (
            completed.stdout
            or ""
        )


        stderr = (
            completed.stderr
            or ""
        ).strip()


        # ----------------------------------------------------
        # Parse curl --write-out metadata
        # ----------------------------------------------------

        parts = stdout.split(
            delimiter
        )


        http_status = None

        final_url = url

        content_type = ""


        if len(parts) >= 4:

            try:

                http_status = int(
                    parts[-3]
                )

            except Exception:

                http_status = None


            final_url = (
                parts[-2]
                or url
            )


            content_type = (
                parts[-1]
                or ""
            )


        # ----------------------------------------------------
        # CURL EXIT 28 = TIMEOUT
        # ----------------------------------------------------

        if (
            completed.returncode
            == 28
        ):

            return {

                "success":
                    False,

                "status":
                    "hard_timeout",

                "curl_returncode":
                    completed.returncode,

                "http_status":
                    http_status,

                "final_url":
                    final_url,

                "error":
                    stderr,
            }


        # ----------------------------------------------------
        # CURL EXIT 63 = MAX FILE SIZE
        # ----------------------------------------------------

        if (
            completed.returncode
            == 63
        ):

            return {

                "success":
                    False,

                "status":
                    "too_large",

                "curl_returncode":
                    completed.returncode,

                "http_status":
                    http_status,

                "final_url":
                    final_url,

                "error":
                    stderr,
            }


        # ----------------------------------------------------
        # OTHER CURL FAILURE
        # ----------------------------------------------------

        if (
            completed.returncode
            != 0
        ):

            return {

                "success":
                    False,

                "status":
                    "curl_error",

                "curl_returncode":
                    completed.returncode,

                "http_status":
                    http_status,

                "final_url":
                    final_url,

                "content_type":
                    content_type,

                "error":
                    stderr,
            }


        # ----------------------------------------------------
        # READ DOWNLOADED FILE
        # ----------------------------------------------------

        temp_file_path = Path(
            temp_path
        )


        if (
            temp_file_path.exists()
        ):

            content = (
                temp_file_path
                .read_bytes()
            )

        else:

            content = b""


        # Extra local size validation.
        if (
            len(content)
            >
            max_bytes
        ):

            return {

                "success":
                    False,

                "status":
                    "too_large",

                "http_status":
                    http_status,

                "final_url":
                    final_url,

                "downloaded_bytes":
                    len(content),
            }


        success = bool(
            http_status is not None
            and
            200 <= http_status < 300
        )


        return {

            "success":
                success,

            "status":
                (
                    "downloaded"
                    if success
                    else "http_error"
                ),

            "http_status":
                http_status,

            "final_url":
                final_url,

            "content":
                content,

            "headers": {
                "Content-Type":
                    content_type
            },

            "content_type":
                content_type,

            "downloaded_bytes":
                len(content),

            "curl_returncode":
                completed.returncode,

            "curl_stderr":
                stderr,
        }


    finally:

        if temp_path:

            try:

                Path(
                    temp_path
                ).unlink(
                    missing_ok=True
                )

            except Exception:

                pass


# ============================================================
# 11. ROBOTS CACHE
# ============================================================

robots_cache = {}

robots_cache_guard = (
    threading.Lock()
)

robots_host_locks = {}


def get_robots_lock(
    host_key
):

    with robots_cache_guard:

        if host_key not in (
            robots_host_locks
        ):

            robots_host_locks[
                host_key
            ] = (
                threading.Lock()
            )


        return (
            robots_host_locks[
                host_key
            ]
        )


def robots_allowed(url):

    if not RESPECT_ROBOTS_TXT:
        return True


    parsed = urlparse(
        url
    )


    host_key = (
        f"{parsed.scheme}://"
        f"{parsed.netloc}"
    )


    # --------------------------------------------------------
    # FAST CACHE PATH
    # --------------------------------------------------------

    with robots_cache_guard:

        cached = (
            robots_cache.get(
                host_key
            )
        )


    if cached is not None:

        return cached.can_fetch(
            USER_AGENT,
            url,
        )


    # --------------------------------------------------------
    # ONE ROBOTS FETCH PER HOST
    # --------------------------------------------------------

    robots_lock = (
        get_robots_lock(
            host_key
        )
    )


    with robots_lock:

        # Another worker might have populated cache while
        # this thread was waiting.
        with robots_cache_guard:

            cached = (
                robots_cache.get(
                    host_key
                )
            )


        if cached is not None:

            return cached.can_fetch(
                USER_AGENT,
                url,
            )


        robots_url = (
            f"{parsed.scheme}://"
            f"{parsed.netloc}"
            f"/robots.txt"
        )


        rp = (
            robotparser
            .RobotFileParser()
        )


        rp.set_url(
            robots_url
        )


        try:

            wait_for_host(
                parsed.hostname
            )


            result = curl_download(

                robots_url,

                hard_timeout=
                    ROBOTS_HARD_TIMEOUT_SECONDS,

                max_bytes=
                    MAX_ROBOTS_BYTES,
            )


            if (
                result.get(
                    "success",
                    False,
                )
                and
                result.get(
                    "content"
                )
            ):

                robots_text = (

                    result[
                        "content"
                    ]
                    .decode(
                        "utf-8",
                        errors="replace",
                    )
                )


                rp.parse(
                    robots_text
                    .splitlines()
                )


            else:

                # Same behavior as previous crawler:
                # inability to retrieve robots.txt does not
                # automatically mean crawling is forbidden.
                rp.parse([])


        except Exception:

            rp.parse([])


        with robots_cache_guard:

            robots_cache[
                host_key
            ] = rp


        return rp.can_fetch(
            USER_AGENT,
            url,
        )


# ============================================================
# 12. PAGE DOWNLOAD WRAPPER
# ============================================================

def download_url(url):

    parsed = urlparse(
        url
    )


    hostname = (
        parsed.hostname
    )


    # --------------------------------------------------------
    # ROBOTS
    # --------------------------------------------------------

    if not robots_allowed(
        url
    ):

        return {

            "success":
                False,

            "status":
                "robots_disallowed",

            "final_url":
                url,

            "error":
                None,
        }


    # --------------------------------------------------------
    # RATE LIMIT
    # --------------------------------------------------------

    wait_for_host(
        hostname
    )


    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    return curl_download(

        url,

        hard_timeout=
            HARD_URL_TIMEOUT_SECONDS,

        max_bytes=
            MAX_DOWNLOAD_BYTES,
    )


# ============================================================
# 13. HTML EXTRACTION
# ============================================================

def extract_html(
    html,
    url,
):

    text = None

    metadata = {}


    # --------------------------------------------------------
    # TRAFILATURA
    # --------------------------------------------------------

    try:

        text = trafilatura.extract(

            html,

            url=url,

            output_format="txt",

            include_comments=False,

            include_tables=True,

            include_links=False,

            favor_recall=True,

            deduplicate=False,
        )


    except Exception:

        text = None


    # --------------------------------------------------------
    # PAGE METADATA
    # --------------------------------------------------------

    try:

        extracted_metadata = (
            trafilatura
            .extract_metadata(
                html,
                default_url=url,
            )
        )


        if (
            extracted_metadata
            is not None
        ):

            metadata = (
                extracted_metadata
                .as_dict()
            )


    except Exception:

        metadata = {}


    # --------------------------------------------------------
    # FALLBACK BEAUTIFULSOUP
    # --------------------------------------------------------

    if (
        text is None
        or
        len(
            text.strip()
        ) < 100
    ):

        try:

            soup = BeautifulSoup(
                html,
                "html.parser",
            )


            for tag in soup(
                [
                    "script",
                    "style",
                    "noscript",
                    "svg",
                ]
            ):

                tag.decompose()


            fallback = soup.get_text(
                "\n",
                strip=True,
            )


            if (
                len(fallback)
                >
                len(
                    text or ""
                )
            ):

                text = fallback


        except Exception:

            pass


    return (
        (
            text
            or ""
        ).strip(),

        metadata,
    )


# ============================================================
# 14. PDF EXTRACTION
# ============================================================

def extract_pdf_text(content):

    document = pymupdf.open(
        stream=content,
        filetype="pdf",
    )


    pages = []


    for page in document:

        pages.append(
            page.get_text(
                "text"
            )
        )


    document.close()


    return "\n\n".join(
        pages
    ).strip()


# ============================================================
# 15. CONTENT TYPE PROCESSING
# ============================================================

def process_content(
    content,
    content_type,
    final_url,
):

    content_type_lower = (
        (
            content_type
            or ""
        )
        .lower()
    )


    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if (
        "application/pdf"
        in content_type_lower
        or
        content[:5]
        == b"%PDF-"
    ):

        text = extract_pdf_text(
            content
        )


        return {

            "kind":
                "pdf",

            "extension":
                ".pdf",

            "text":
                text,

            "metadata":
                {},
        }


    # --------------------------------------------------------
    # HTML
    # --------------------------------------------------------

    if (
        "text/html"
        in content_type_lower
        or
        "application/xhtml+xml"
        in content_type_lower
        or
        b"<html"
        in content[:5000].lower()
    ):

        html = content.decode(
            "utf-8",
            errors="replace",
        )


        text, metadata = (
            extract_html(
                html,
                final_url,
            )
        )


        return {

            "kind":
                "html",

            "extension":
                ".html",

            "text":
                text,

            "metadata":
                metadata,
        }


    # --------------------------------------------------------
    # PLAIN TEXT
    # --------------------------------------------------------

    if (
        "text/"
        in content_type_lower
    ):

        text = content.decode(
            "utf-8",
            errors="replace",
        )


        return {

            "kind":
                "text",

            "extension":
                ".txt",

            "text":
                text.strip(),

            "metadata":
                {},
        }


    # --------------------------------------------------------
    # UNKNOWN BINARY
    # --------------------------------------------------------

    return {

        "kind":
            "binary",

        "extension":
            ".bin",

        "text":
            "",

        "metadata":
            {},
    }


# ============================================================
# 16. METADATA / RESUME
# ============================================================

def metadata_path(
    source_id
):

    return (
        METADATA_DIR
        / f"{source_id}.json"
    )


def read_existing_metadata(
    source_id
):

    path = metadata_path(
        source_id
    )


    if not path.exists():

        return None


    try:

        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )


    except Exception:

        # Corrupt / interrupted JSON:
        # treat as pending.
        return None


def existing_is_complete(
    record
):

    if record is None:
        return False


    # --------------------------------------------------------
    # Current behavior:
    # every prior attempt is considered completed.
    # --------------------------------------------------------

    if not RETRY_FAILED_EXISTING:
        return True


    # --------------------------------------------------------
    # Optional future behavior:
    # retry anything except actual successful extraction.
    # --------------------------------------------------------

    return (
        record.get(
            "extraction_status"
        )
        == "success"
    )


# ============================================================
# 17. ATOMIC JSON WRITE
# ============================================================

def atomic_write_json(
    path,
    record,
):

    temp_path = (
        path.parent
        /
        (
            path.name
            + ".tmp-"
            + str(
                threading.get_ident()
            )
        )
    )


    temp_path.write_text(

        json.dumps(
            record,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),

        encoding="utf-8",
    )


    os.replace(
        temp_path,
        path,
    )


# ============================================================
# 18. PROCESS ONE SOURCE
# ============================================================

def process_source(
    source_id,
    url,
):

    record = {

        "source_id":
            source_id,

        "normalized_url":
            url,

        "fetch_timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "registrable_domain":
            registrable_domain(
                url
            ),
    }


    try:

        result = download_url(
            url
        )


        record.update({

            key:
                value

            for key, value
            in result.items()

            if key != "content"
        })


        # ----------------------------------------------------
        # DOWNLOAD FAILED
        # ----------------------------------------------------

        if not result.get(
            "success",
            False,
        ):

            record[
                "extraction_status"
            ] = (
                "not_extracted"
            )


            record[
                "text_chars"
            ] = 0


            record[
                "text_words"
            ] = 0


        # ----------------------------------------------------
        # DOWNLOAD SUCCESS
        # ----------------------------------------------------

        else:

            content = (
                result[
                    "content"
                ]
            )


            headers = (
                result.get(
                    "headers",
                    {}
                )
            )


            content_type = (
                headers.get(
                    "Content-Type",
                    ""
                )
            )


            processed = process_content(

                content,

                content_type,

                result[
                    "final_url"
                ],
            )


            kind = (
                processed[
                    "kind"
                ]
            )


            extension = (
                processed[
                    "extension"
                ]
            )


            text = (
                processed[
                    "text"
                ]
            )


            page_metadata = (
                processed.get(
                    "metadata",
                    {}
                )
                or {}
            )


            # ------------------------------------------------
            # RAW CONTENT
            # ------------------------------------------------

            raw_path = (
                RAW_DIR
                /
                (
                    source_id
                    + extension
                )
            )


            raw_path.write_bytes(
                content
            )


            # ------------------------------------------------
            # EXTRACTED TEXT
            # ------------------------------------------------

            text_path = (
                TEXT_DIR
                /
                f"{source_id}.txt"
            )


            text_path.write_text(
                text,
                encoding="utf-8",
            )


            # ------------------------------------------------
            # METADATA
            # ------------------------------------------------

            record.update({

                "content_kind":
                    kind,

                "content_type":
                    content_type,

                "raw_path":
                    str(
                        raw_path
                    ),

                "text_path":
                    str(
                        text_path
                    ),

                "text_chars":
                    len(text),

                "text_words":
                    len(
                        text.split()
                    ),

                "extraction_status":
                    (
                        "success"
                        if text.strip()
                        else "empty_text"
                    ),

                "page_title":
                    page_metadata.get(
                        "title"
                    ),

                "page_author":
                    page_metadata.get(
                        "author"
                    ),

                "page_date":
                    page_metadata.get(
                        "date"
                    ),

                "site_name":
                    page_metadata.get(
                        "sitename"
                    ),

                "canonical_url":
                    page_metadata.get(
                        "url"
                    ),
            })


    except Exception as exc:

        record.update({

            "success":
                False,

            "status":
                "exception",

            "error_type":
                type(exc).__name__,

            "error":
                str(exc),

            "traceback":
                traceback.format_exc(),

            "extraction_status":
                "failed",

            "text_chars":
                0,

            "text_words":
                0,
        })


    # --------------------------------------------------------
    # SAVE RESULT IMMEDIATELY
    #
    # This is why resume works even if the full script dies.
    # --------------------------------------------------------

    atomic_write_json(

        metadata_path(
            source_id
        ),

        record,
    )


    return record


# ============================================================
# 19. RESUME: FIND COMPLETED / PENDING URLS
# ============================================================

existing_records = {}

pending_sources = []


for _, source in (
    unique_sources.iterrows()
):

    source_id = (
        source[
            "source_id"
        ]
    )


    url = (
        source[
            "normalized_url"
        ]
    )


    existing = (
        read_existing_metadata(
            source_id
        )
    )


    if existing_is_complete(
        existing
    ):

        existing_records[
            source_id
        ] = existing


    else:

        pending_sources.append(
            (
                source_id,
                url,
            )
        )


print()
print("=" * 80)
print("RESUME STATUS")
print("=" * 80)


print(
    f"\nUnique URLs total: "
    f"{len(unique_sources):,}"
)


print(
    f"Already processed: "
    f"{len(existing_records):,}"
)


print(
    f"Still pending: "
    f"{len(pending_sources):,}"
)


print(
    f"Workers: "
    f"{MAX_WORKERS}"
)


print(
    f"Per-host interval: "
    f"{DELAY_PER_HOST_SECONDS}s"
)


print(
    f"Hard URL timeout: "
    f"{HARD_URL_TIMEOUT_SECONDS}s"
)


if pending_sources:

    print(
        "\nPending URLs:"
    )


    for source_id, url in (
        pending_sources
    ):

        print(
            f"  {source_id} | "
            f"{url}"
        )


# ============================================================
# 20. PARALLEL COLLECTION
# ============================================================

print()
print("=" * 80)
print("PARALLEL SOURCE COLLECTION")
print("=" * 80)


new_success = 0

new_failed = 0

completed_count = 0


if pending_sources:

    executor = ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    )


    future_to_source = {

        executor.submit(
            process_source,
            source_id,
            url,
        ):
        (
            source_id,
            url,
        )

        for source_id, url
        in pending_sources
    }


    try:

        for future in as_completed(
            future_to_source
        ):

            source_id, url = (
                future_to_source[
                    future
                ]
            )


            completed_count += 1


            try:

                record = (
                    future.result()
                )


                if (
                    record.get(
                        "extraction_status"
                    )
                    == "success"
                ):

                    new_success += 1

                    marker = "OK"


                else:

                    new_failed += 1


                    marker = (
                        record.get(
                            "status"
                        )
                        or
                        record.get(
                            "extraction_status"
                        )
                        or
                        "FAILED"
                    )


            except Exception as exc:

                new_failed += 1


                marker = (
                    "WORKER_EXCEPTION:"
                    f"{type(exc).__name__}"
                )


            print(
                f"[{completed_count:,}/"
                f"{len(pending_sources):,}] "
                f"{marker} | "
                f"{url[:120]}"
            )


    except KeyboardInterrupt:

        print()
        print(
            "KeyboardInterrupt received."
        )

        print(
            "Cancelling jobs that have not "
            "started yet..."
        )


        for future in (
            future_to_source
        ):

            future.cancel()


        executor.shutdown(
            wait=False,
            cancel_futures=True,
        )


        print(
            "Already completed metadata remains saved."
        )


        raise


    else:

        executor.shutdown(
            wait=True
        )


else:

    print(
        "\nNothing pending."
    )

    print(
        "All unique URLs already have metadata."
    )


# ============================================================
# 21. REBUILD COMPLETE MANIFEST FROM DISK
#
# Combines:
#   - old run metadata
#   - new run metadata
# ============================================================

print()
print("=" * 80)
print("REBUILDING COMPLETE MANIFEST")
print("=" * 80)


manifest_rows = []

missing_after_run = []


for _, source in (
    unique_sources.iterrows()
):

    source_id = (
        source[
            "source_id"
        ]
    )


    url = (
        source[
            "normalized_url"
        ]
    )


    record = (
        read_existing_metadata(
            source_id
        )
    )


    if record is None:

        missing_after_run.append({

            "source_id":
                source_id,

            "normalized_url":
                url,
        })


        continue


    manifest_rows.append(
        record
    )


manifest = pd.DataFrame(
    manifest_rows
)


if not manifest.empty:

    manifest = (
        manifest
        .drop_duplicates(
            subset=[
                "source_id"
            ],
            keep="last",
        )
        .sort_values(
            "source_id"
        )
        .reset_index(
            drop=True
        )
    )


manifest.to_csv(
    CORPUS_DIR
    / "manifest.csv",
    index=False,
)


# ============================================================
# 22. URL USAGE + FETCH METADATA
# ============================================================

if not manifest.empty:

    usage_enriched = (
        usage.merge(

            manifest,

            on=[
                "source_id",
                "normalized_url",
            ],

            how="left",
        )
    )

else:

    usage_enriched = (
        usage.copy()
    )


usage_enriched.to_csv(
    CORPUS_DIR
    / "url_usage_enriched.csv",
    index=False,
)


# ============================================================
# 23. CORPUS JSONL
#
# One unique URL per line.
# Includes extracted text.
# ============================================================

jsonl_path = (
    CORPUS_DIR
    / "corpus.jsonl"
)


with jsonl_path.open(
    "w",
    encoding="utf-8",
) as output:

    for _, row in (
        manifest.iterrows()
    ):

        record = (
            row
            .where(
                pd.notna(row),
                None,
            )
            .to_dict()
        )


        text_path = (
            record.get(
                "text_path"
            )
        )


        if (
            isinstance(
                text_path,
                str,
            )
            and text_path
            and Path(
                text_path
            ).exists()
        ):

            record[
                "extracted_text"
            ] = (
                Path(
                    text_path
                )
                .read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            )


        else:

            record[
                "extracted_text"
            ] = ""


        output.write(

            json.dumps(
                record,
                ensure_ascii=False,
                default=str,
            )

            + "\n"
        )


# ============================================================
# 24. COLLECTION QUALITY SUMMARY
# ============================================================

total_sources = len(
    unique_sources
)


if manifest.empty:

    successful_downloads = 0

    successful_extractions = 0

    empty_extractions = 0

    robots_blocked = 0

    hard_timeouts = 0

    http_errors = 0

    curl_errors = 0

    exceptions = 0


else:

    success_series = (
        manifest.get(
            "success",
            pd.Series(
                False,
                index=manifest.index,
            )
        )
        .fillna(False)
        .astype(bool)
    )


    status_series = (
        manifest.get(
            "status",
            pd.Series(
                "",
                index=manifest.index,
            )
        )
        .fillna("")
    )


    extraction_series = (
        manifest.get(
            "extraction_status",
            pd.Series(
                "",
                index=manifest.index,
            )
        )
        .fillna("")
    )


    successful_downloads = int(
        success_series.sum()
    )


    successful_extractions = int(
        (
            extraction_series
            == "success"
        ).sum()
    )


    empty_extractions = int(
        (
            extraction_series
            == "empty_text"
        ).sum()
    )


    robots_blocked = int(
        (
            status_series
            == "robots_disallowed"
        ).sum()
    )


    hard_timeouts = int(
        (
            status_series
            == "hard_timeout"
        ).sum()
    )


    http_errors = int(
        (
            status_series
            == "http_error"
        ).sum()
    )


    curl_errors = int(
        (
            status_series
            == "curl_error"
        ).sum()
    )


    exceptions = int(
        (
            status_series
            == "exception"
        ).sum()
    )


summary = {

    "collection_version":
        COLLECTION_VERSION,

    "report_timestamp_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "aio_collection_directory":
        str(
            AIO_COLLECTION_DIR
        ),

    "source_corpus_directory":
        str(
            CORPUS_DIR
        ),

    "total_source_occurrences":
        len(
            usage
        ),

    "unique_urls":
        total_sources,

    "already_processed_before_run":
        len(
            existing_records
        ),

    "scheduled_this_run":
        len(
            pending_sources
        ),

    "new_successful_extractions":
        new_success,

    "new_failed_or_incomplete":
        new_failed,

    "metadata_records_after_run":
        len(
            manifest
        ),

    "missing_metadata_after_run":
        len(
            missing_after_run
        ),

    "successful_downloads_total":
        successful_downloads,

    "successful_text_extractions_total":
        successful_extractions,

    "download_success_rate":
        (
            successful_downloads
            / total_sources
            if total_sources
            else None
        ),

    "text_extraction_rate":
        (
            successful_extractions
            / total_sources
            if total_sources
            else None
        ),

    "empty_text_extractions":
        empty_extractions,

    "robots_blocked":
        robots_blocked,

    "hard_timeouts":
        hard_timeouts,

    "http_errors":
        http_errors,

    "curl_errors":
        curl_errors,

    "exceptions":
        exceptions,

    "max_workers":
        MAX_WORKERS,

    "delay_per_host_seconds":
        DELAY_PER_HOST_SECONDS,

    "hard_url_timeout_seconds":
        HARD_URL_TIMEOUT_SECONDS,

    "robots_hard_timeout_seconds":
        ROBOTS_HARD_TIMEOUT_SECONDS,

    "retry_failed_existing":
        RETRY_FAILED_EXISTING,

    "user_agent":
        USER_AGENT,

    "max_download_bytes":
        MAX_DOWNLOAD_BYTES,
}


(
    RESULTS_DIR
    / "collection_summary.json"
).write_text(

    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),

    encoding="utf-8",
)


# ============================================================
# 25. FAILED / INCOMPLETE SOURCES
# ============================================================

if not manifest.empty:

    failures = manifest[
        manifest[
            "extraction_status"
        ]
        .fillna("")
        != "success"
    ].copy()


else:

    failures = pd.DataFrame()


failures.to_csv(
    RESULTS_DIR
    / "failed_or_incomplete_sources.csv",
    index=False,
)


# ============================================================
# 26. STILL MISSING METADATA
# ============================================================

pd.DataFrame(
    missing_after_run
).to_csv(
    RESULTS_DIR
    / "missing_after_run.csv",
    index=False,
)


# ============================================================
# 27. COLLECTION QUALITY BY DOMAIN
# ============================================================

if (
    not manifest.empty
    and
    "registrable_domain"
    in manifest.columns
):

    domain_quality = (
        manifest
        .groupby(
            "registrable_domain",
            dropna=False,
        )
        .agg(

            n_sources=(
                "source_id",
                "size",
            ),

            n_success=(
                "extraction_status",
                lambda values:
                    int(
                        (
                            values
                            == "success"
                        ).sum()
                    ),
            ),

            n_timeout=(
                "status",
                lambda values:
                    int(
                        (
                            values
                            == "hard_timeout"
                        ).sum()
                    ),
            ),

            n_robots_blocked=(
                "status",
                lambda values:
                    int(
                        (
                            values
                            == "robots_disallowed"
                        ).sum()
                    ),
            ),
        )
        .reset_index()
    )


    domain_quality[
        "success_rate"
    ] = (
        domain_quality[
            "n_success"
        ]
        /
        domain_quality[
            "n_sources"
        ]
    )


    domain_quality.to_csv(
        RESULTS_DIR
        / "collection_quality_by_domain.csv",
        index=False,
    )


# ============================================================
# 28. FINAL REPORT
# ============================================================

print()
print("=" * 80)
print("SOURCE COLLECTION FINISHED")
print("=" * 80)


print(
    f"\nSource occurrences: "
    f"{len(usage):,}"
)


print(
    f"Unique URLs: "
    f"{total_sources:,}"
)


print(
    f"\nAlready present before this run: "
    f"{len(existing_records):,}"
)


print(
    f"Scheduled this run: "
    f"{len(pending_sources):,}"
)


print(
    f"New successful extractions: "
    f"{new_success:,}"
)


print(
    f"New failed/incomplete: "
    f"{new_failed:,}"
)


print()
print(
    f"Successful downloads total: "
    f"{successful_downloads:,}"
    f"/{total_sources:,}"
)


print(
    f"Successful text extractions total: "
    f"{successful_extractions:,}"
    f"/{total_sources:,}"
)


if total_sources:

    print(
        f"Text extraction coverage: "
        f"{successful_extractions / total_sources:.2%}"
    )


print()
print(
    f"Hard timeouts: "
    f"{hard_timeouts:,}"
)


print(
    f"Robots blocked: "
    f"{robots_blocked:,}"
)


print(
    f"HTTP errors: "
    f"{http_errors:,}"
)


print(
    f"curl errors: "
    f"{curl_errors:,}"
)


print(
    f"Exceptions: "
    f"{exceptions:,}"
)


print(
    f"Empty text extractions: "
    f"{empty_extractions:,}"
)


print(
    f"Missing metadata after run: "
    f"{len(missing_after_run):,}"
)


print()
print(
    "Corpus directory:"
)

print(
    CORPUS_DIR
)


print()
print(
    "Reports directory:"
)

print(
    RESULTS_DIR
)


print()
print(
    "Main files:"
)


for filename in [

    "manifest.csv",

    "url_usage.csv",

    "url_usage_enriched.csv",

    "corpus.jsonl",

]:

    print(
        f"  {CORPUS_DIR / filename}"
    )


for filename in [

    "collection_summary.json",

    "failed_or_incomplete_sources.csv",

    "missing_after_run.csv",

    "collection_quality_by_domain.csv",

]:

    print(
        f"  {RESULTS_DIR / filename}"
    )


print()
print(
    "Done."
)