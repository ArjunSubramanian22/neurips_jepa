#!/usr/bin/env python3
"""Recursive anonymity / secret scanner for this anonymous artifact.

Scans the entire tree (not just results/). Prints file, line, and reason for
every unresolved hit. Exit status is 1 if any unresolved leak remains.

Bibliography author names, NeurIPS style-file credits, and third-party license
notices are whitelisted. Submitter identifiers are matched from encoded
fragments so this file does not itself contain those strings.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
    ".mplconfig",
}

TEXT_SUFFIXES = {
    ".py",
    ".tex",
    ".bib",
    ".bbl",
    ".md",
    ".txt",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".csv",
    ".tsv",
    ".sh",
    ".toml",
    ".ini",
    ".cfg",
    ".log",
    ".sty",
}

# Built from hex so this source does not contain submitter identifiers.
_FORBIDDEN = [
    bytes.fromhex("41726a756e").decode(),  # given name used in local paths
    bytes.fromhex("41726a756e53756272616d616e69616e3232").decode(),  # hosting username
    bytes.fromhex("61726a756e5f73").decode(),  # local account prefix
    bytes.fromhex("454c4162756e64696f").decode(),  # git author string
    bytes.fromhex("61726a756e5f73406d69742e656475").decode(),  # institutional email
]

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
USERS_PATH_RE = re.compile(r"/Users/[A-Za-z0-9._-]+")
HOME_PATH_RE = re.compile(r"/home/[A-Za-z0-9._-]+")
WIN_USER_RE = re.compile(r"[A-Za-z]:\\Users\\[A-Za-z0-9._-]+", re.I)
WIN_DOC_RE = re.compile(r"\\(Users|Documents|Desktop|Downloads)\\", re.I)
GIT_SSH_RE = re.compile(r"git@github\.com:")
GIT_HTTPS_RE = re.compile(r"https?://github\.com/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+")
GITHUB_USER_RE = re.compile(r"github\.com/[A-Za-z0-9._-]+")
AUTHOR_FIELD_RE = re.compile(
    r"""(?:^|[\s,{])(?:author|authors|affiliation|institution|university|school|orcid)\s*[=:]\s*["']?([^\n"'\\}]+)""",
    re.I,
)
HOSTNAME_FIELD_RE = re.compile(
    r"""(?:hostname|username|user_name|created_by|owner)\s*[=:]\s*["']?([^\n"'\\}]+)""",
    re.I,
)
PERSONAL_DIR_RE = re.compile(
    r"(?:/Desktop/|/Documents/|/Downloads/|/OneDrive/|/iCloud/)",
    re.I,
)
ABS_EXEC_RE = re.compile(r"/Library/Frameworks/Python|/opt/homebrew/Cellar")

SECRET_PATTERNS = [
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}")),
    ("openai_key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.I)),
    ("pem_key", re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")),
    ("password_assign", re.compile(r"""(?:password|passwd|api_key|secret)\s*[=:]\s*['"][^'"]{8,}['"]""", re.I)),
]

THIRD_PARTY_EMAILS: set[str] = set()

ANON_AUTHOR_OK = re.compile(r"anonymous author", re.I)
WORKSHOP_HOST_OK = re.compile(r"physworld-org\.github\.io|media\.neurips\.cc")


def _is_text(path: Path) -> bool:
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    if path.name in {".gitignore", "LICENSE", "Makefile"}:
        return True
    return False


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _whitelisted(path: Path, line: str, kind: str) -> bool:
    rel = _rel(path)
    if rel in {"scripts/check_anonymity.py", "LICENSE_AUDIT.md", "ANONYMITY_AUDIT.md"}:
        return True
    if kind == "author_field" and ANON_AUTHOR_OK.search(line):
        return True
    if kind in {"github_url", "github_user"} and WORKSHOP_HOST_OK.search(line):
        return True
    if kind == "email":
        for addr in EMAIL_RE.findall(line):
            if addr.lower() not in THIRD_PARTY_EMAILS and "wustl.edu" not in addr.lower():
                return False
        return True
    return False


def scan_line(path: Path, line_no: int, line: str) -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    checks = [
        ("email", EMAIL_RE),
        ("home_path", USERS_PATH_RE),
        ("unix_home", HOME_PATH_RE),
        ("windows_user_path", WIN_USER_RE),
        ("windows_personal_dir", WIN_DOC_RE),
        ("git_ssh_remote", GIT_SSH_RE),
        ("github_url", GIT_HTTPS_RE),
        ("github_user", GITHUB_USER_RE),
        ("personal_directory", PERSONAL_DIR_RE),
        ("absolute_framework_path", ABS_EXEC_RE),
        ("author_field", AUTHOR_FIELD_RE),
        ("hostname_user_field", HOSTNAME_FIELD_RE),
    ]
    for kind, pat in checks:
        if not pat.search(line):
            continue
        if _whitelisted(path, line, kind):
            continue
        if kind == "author_field":
            match = AUTHOR_FIELD_RE.search(line)
            value = (match.group(1) if match else "").strip()
            if not value or ANON_AUTHOR_OK.search(value) or value.lower() in {
                "true",
                "false",
                "none",
                "null",
            }:
                continue
            # Bibliography / citation context is handled by whitelist; remaining
            # author= in code/config is a leak unless it is clearly anonymous.
            if path.suffix.lower() in {".bib", ".bbl"}:
                continue
        if kind == "hostname_user_field":
            match = HOSTNAME_FIELD_RE.search(line)
            value = (match.group(1) if match else "").strip().lower()
            if value in {"true", "false", "none", "null", "", "anonymous"}:
                continue
            if "no hostname" in line.lower() or "usernames omitted" in line.lower():
                continue
        hits.append(( _rel(path), line_no, kind))
    for token in _FORBIDDEN:
        if token and token in line:
            if _whitelisted(path, line, "forbidden_identifier"):
                continue
            hits.append((_rel(path), line_no, "forbidden_identifier"))
            break
    for secret_kind, pat in SECRET_PATTERNS:
        if pat.search(line):
            hits.append((_rel(path), line_no, f"secret:{secret_kind}"))
    return hits


def scan_binary_meta(path: Path) -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    data = path.read_bytes()
    # Cheap binary identity scan; do not print matched bytes.
    for token in _FORBIDDEN:
        if token.encode() in data:
            hits.append((_rel(path), 0, "forbidden_identifier_binary"))
    if b"/Users/" in data:
        hits.append((_rel(path), 0, "home_path_binary"))
    if b"/home/" in data:
        hits.append((_rel(path), 0, "unix_home_binary"))
    if path.suffix.lower() == ".pdf":
        text = data.decode("latin-1", errors="ignore")
        for field in ("/Author", "/Creator", "/Producer", "/Title", "/Subject"):
            idx = text.find(field)
            if idx < 0:
                continue
            snippet = text[idx : idx + 120]
            if field in {"/Creator", "/Producer"} and (
                "TeX" in snippet or "pdfTeX" in snippet or "LaTeX" in snippet or "tectonic" in snippet.lower()
            ):
                continue
            for token in _FORBIDDEN:
                if token in snippet:
                    hits.append((_rel(path), 0, f"pdf_metadata:{field[1:]}"))
    return hits


def iter_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        files.append(path)
    return files


def main() -> int:
    files = iter_files()
    hits: list[tuple[str, int, str]] = []
    n_text = 0
    for path in files:
        if _is_text(path):
            n_text += 1
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                hits.extend(scan_line(path, i, line))
        else:
            hits.extend(scan_binary_meta(path))
    # Deduplicate
    uniq = sorted(set(hits))
    for file, line_no, reason in uniq:
        loc = f"{file}:{line_no}" if line_no else file
        print(f"{loc}  {reason}")
    print(f"scanned_files={len(files)} text_files={n_text} hits={len(uniq)}")
    if uniq:
        print("ANONYMITY: FAIL")
        return 1
    print("ANONYMITY: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
