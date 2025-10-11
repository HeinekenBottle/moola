#!/usr/bin/env python3
import json, os, re, sys, subprocess, shutil
from pathlib import Path

def deny(msg: str, code: int = 2):
    sys.stderr.write(f"deny: {msg}\n")
    sys.exit(code)  # 2 blocks the tool call per Claude Code spec

# 0) Parse JSON from stdin
try:
    data = json.load(sys.stdin)
except Exception:
    deny("invalid JSON input")

tool_input = data.get("tool_input", {}) or {}

# 1) Must be inside Moola repo
try:
    repo = Path(
        subprocess.check_output(["git","rev-parse","--show-toplevel"], stderr=subprocess.DEVNULL)
        .decode().strip()
    )
except Exception:
    repo = Path.cwd()

if repo.name != "moola":
    deny("outside Moola repo")

# 2) Require python3 usage in commands
cmd = tool_input.get("command") or tool_input.get("bash") or tool_input.get("cmd") or ""
if re.search(r"\bpython\b(?!3)", cmd):
    deny("use 'python3', not 'python'")

# 3) Extract paths and validate they stay within repo
paths = []
for k in ("file_path","path","dest","src"):
    v = tool_input.get(k)
    if isinstance(v, str):
        paths.append(v)
for k in ("file_paths","paths"):
    v = tool_input.get(k) or []
    if isinstance(v, list):
        paths.extend([p for p in v if isinstance(p, str)])

# Normalize and check
banned = re.compile(r"(^/|\.{2}/|/\.{2}/)|(/\.git/|^\.git/|/\.ssh/|^\.ssh/|/secrets|^secrets|/\.env|^\.env)")
repo_abs = repo.resolve()

for p in paths:
    if banned.search(p):
        deny(f"unsafe path: {p}")
    ap = (repo_abs / p).resolve() if not os.path.isabs(p) else Path(p).resolve()
    try:
        common = os.path.commonpath([str(repo_abs), str(ap)])
    except Exception:
        deny(f"bad path: {p}")
    if Path(common) != repo_abs:
        deny(f"path escapes repo: {p}")

# 4) Fast hygiene (optional, silent if missing)
if shutil.which("ruff"):
    try:
        subprocess.check_call(["ruff","--quiet","."], cwd=str(repo_abs), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        deny("ruff lint failed")

# 5) Minimal data contract ping if file exists (non-blocking on absence)
pp = repo_abs / "data" / "processed" / "train.parquet"
if pp.exists():
    try:
        import pandas as pd
        df = pd.read_parquet(pp)
        req = {"window_id","label","features"}
        if not req.issubset(df.columns):
            deny("train.parquet schema mismatch")
    except Exception:
        deny("cannot read train.parquet")

# All checks passed
sys.exit(0)
