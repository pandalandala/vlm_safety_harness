#!/usr/bin/env python3
"""
Prefetch Hugging Face models listed in a Markdown file into the local HF cache.

Examples:
    python scripts/prefetch_hf_models.py
    python scripts/prefetch_hf_models.py --hf-home /mnt2/xuran_hdd/cache/hf
    python scripts/prefetch_hf_models.py --cache-dir /mnt2/xuran_hdd/cache/hf/hub
    python scripts/prefetch_hf_models.py --section "MIRage Checkpoints" --dry-run
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import sys
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse


DEFAULT_MODELS_MD = Path(__file__).resolve().parent.parent / "docs" / "Models_List.md"
HF_URL_PREFIX = "https://huggingface.co/"
HF_REPO_PATTERN = re.compile(r"https://huggingface\.co/[^\s)]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download all Hugging Face models referenced in a Markdown file into the HF cache."
    )
    parser.add_argument(
        "--models-md",
        type=Path,
        default=DEFAULT_MODELS_MD,
        help=f"Path to the Markdown model list (default: {DEFAULT_MODELS_MD})",
    )
    parser.add_argument(
        "--section",
        action="append",
        default=[],
        help="Only download models from the given Markdown section title. Repeatable.",
    )
    parser.add_argument(
        "--hf-home",
        default=None,
        help="Optional HF_HOME value. Useful when you want the cache under a specific root.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional cache_dir passed to huggingface_hub.snapshot_download.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. If omitted, use HF_TOKEN/HUGGINGFACE_HUB_TOKEN from env.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=2,
        help="How many model repos to download in parallel. Default: 2.",
    )
    parser.add_argument(
        "--file-workers",
        type=int,
        default=8,
        help="How many files to download in parallel inside each repo. Passed to snapshot_download(max_workers=...).",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=[],
        help="Optional allow_patterns passed to snapshot_download. Repeatable.",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=[],
        help="Optional ignore_patterns passed to snapshot_download. Repeatable.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip download if the target repo is already present in the chosen cache directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the repos that would be downloaded.",
    )
    return parser.parse_args()


def normalize_repo_id(url: str) -> str:
    if not url.startswith(HF_URL_PREFIX):
        raise ValueError(f"Not a Hugging Face URL: {url}")

    path = urlparse(url).path.strip("/")
    parts = [part for part in path.split("/") if part]
    if len(parts) < 2:
        raise ValueError(f"Could not parse repo id from URL: {url}")

    repo_owner, repo_name = parts[0], parts[1]
    return f"{repo_owner}/{repo_name}"


def extract_sectioned_repo_ids(markdown_text: str) -> dict[str, list[str]]:
    section_to_repos: dict[str, list[str]] = {}
    current_section = "ROOT"

    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("#"):
            current_section = line.lstrip("#").strip()
            section_to_repos.setdefault(current_section, [])
            continue

        match = HF_REPO_PATTERN.search(line)
        if not match:
            continue

        repo_id = normalize_repo_id(match.group(0))
        section_to_repos.setdefault(current_section, []).append(repo_id)

    # Deduplicate while preserving order inside each section.
    deduped: dict[str, list[str]] = {}
    for section, repo_ids in section_to_repos.items():
        seen: set[str] = set()
        deduped[section] = []
        for repo_id in repo_ids:
            if repo_id in seen:
                continue
            seen.add(repo_id)
            deduped[section].append(repo_id)
    return deduped


def select_repo_ids(section_to_repos: dict[str, list[str]], sections: Iterable[str]) -> list[str]:
    wanted_sections = [section.strip() for section in sections if section.strip()]
    if not wanted_sections:
        selected_sections = list(section_to_repos.keys())
    else:
        missing = [section for section in wanted_sections if section not in section_to_repos]
        if missing:
            available = ", ".join(section_to_repos.keys())
            raise ValueError(f"Unknown section(s): {missing}. Available sections: {available}")
        selected_sections = wanted_sections

    seen: set[str] = set()
    selected_repo_ids: list[str] = []
    for section in selected_sections:
        for repo_id in section_to_repos.get(section, []):
            if repo_id in seen:
                continue
            seen.add(repo_id)
            selected_repo_ids.append(repo_id)
    return selected_repo_ids


def repo_exists_in_cache(repo_id: str, cache_dir: str | None) -> bool:
    if not cache_dir:
        return False

    repo_dir = repo_id.replace("/", "--")
    snapshots_root = Path(cache_dir) / f"models--{repo_dir}" / "snapshots"
    if not snapshots_root.exists():
        return False

    return any(child.is_dir() for child in snapshots_root.iterdir())


def resolve_effective_cache_dir(hf_home: str | None, cache_dir: str | None) -> Path:
    if cache_dir:
        return Path(cache_dir).expanduser().resolve()

    if hf_home:
        hf_home_path = Path(hf_home).expanduser().resolve()
        if hf_home_path.name == "hub":
            return hf_home_path
        return (hf_home_path / "hub").resolve()

    return (Path.home() / ".cache" / "huggingface" / "hub").resolve()


def download_one_repo(
    repo_id: str,
    *,
    cache_dir: str | None,
    token: str | None,
    allow_patterns: list[str],
    ignore_patterns: list[str],
    file_workers: int,
) -> tuple[str, str]:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        cache_dir=cache_dir,
        token=token,
        allow_patterns=allow_patterns or None,
        ignore_patterns=ignore_patterns or None,
        max_workers=file_workers,
    )
    return repo_id, "done"


def main() -> int:
    args = parse_args()

    if args.jobs < 1:
        print("[error] --jobs must be >= 1", file=sys.stderr)
        return 1

    if args.file_workers < 1:
        print("[error] --file-workers must be >= 1", file=sys.stderr)
        return 1

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    effective_cache_dir = resolve_effective_cache_dir(os.environ.get("HF_HOME"), args.cache_dir)

    models_md = args.models_md.resolve()
    if not models_md.exists():
        print(f"[error] Model list not found: {models_md}", file=sys.stderr)
        return 1

    section_to_repos = extract_sectioned_repo_ids(models_md.read_text(encoding="utf-8"))
    repo_ids = select_repo_ids(section_to_repos, args.section)

    if not repo_ids:
        print(f"[warn] No Hugging Face model URLs found in {models_md}")
        return 0

    print(f"[info] model list: {models_md}")
    print(f"[info] sections: {', '.join(args.section) if args.section else 'ALL'}")
    print(f"[info] repos: {len(repo_ids)}")
    print(f"[info] HF_HOME: {os.environ.get('HF_HOME', '<default>')}")
    print(f"[info] cache_dir: {effective_cache_dir}")
    print(f"[info] jobs: {args.jobs}")
    print(f"[info] file_workers: {args.file_workers}")

    for idx, repo_id in enumerate(repo_ids, start=1):
        print(f"  {idx:02d}. {repo_id}")

    if args.dry_run:
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "[error] Missing dependency: huggingface_hub. Install it first, for example:\n"
            "        pip install huggingface_hub",
            file=sys.stderr,
        )
        return 1

    pending_repo_ids: list[str] = []
    for repo_id in repo_ids:
        if args.skip_existing and repo_exists_in_cache(repo_id, str(effective_cache_dir)):
            print(f"[skip] {repo_id} already exists in cache_dir={effective_cache_dir}")
            continue
        pending_repo_ids.append(repo_id)

    if not pending_repo_ids:
        print("\n[summary] Nothing to download.")
        return 0

    failures: list[tuple[str, str]] = []
    if args.jobs == 1:
        for repo_id in pending_repo_ids:
            print(f"[download] {repo_id}")
            try:
                download_one_repo(
                    repo_id,
                    cache_dir=args.cache_dir,
                    token=token,
                    allow_patterns=args.allow_pattern,
                    ignore_patterns=args.ignore_pattern,
                    file_workers=args.file_workers,
                )
            except Exception as exc:  # noqa: BLE001
                failures.append((repo_id, str(exc)))
                print(f"[failed] {repo_id}: {exc}", file=sys.stderr)
                continue

            print(f"[done] {repo_id}")
    else:
        for repo_id in pending_repo_ids:
            print(f"[queue] {repo_id}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            future_to_repo = {
                executor.submit(
                    download_one_repo,
                    repo_id,
                    cache_dir=args.cache_dir,
                    token=token,
                    allow_patterns=args.allow_pattern,
                    ignore_patterns=args.ignore_pattern,
                    file_workers=args.file_workers,
                ): repo_id
                for repo_id in pending_repo_ids
            }

            for future in concurrent.futures.as_completed(future_to_repo):
                repo_id = future_to_repo[future]
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001
                    failures.append((repo_id, str(exc)))
                    print(f"[failed] {repo_id}: {exc}", file=sys.stderr)
                    continue

                print(f"[done] {repo_id}")

    if failures:
        print("\n[summary] Some downloads failed:", file=sys.stderr)
        for repo_id, error_msg in failures:
            print(f"  - {repo_id}: {error_msg}", file=sys.stderr)
        return 2

    print("\n[summary] All requested model snapshots were downloaded successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
