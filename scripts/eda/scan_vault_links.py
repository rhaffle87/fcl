"""
Scan all markdown wikilinks and PDF references across the Obsidian vault.
Usage:
    python scan_vault_links.py [--vault E:/Projects/thesis]
"""

import argparse
import pathlib
import re


def scan_vault(vault_dir: str = "E:/Projects/thesis"):
    root = pathlib.Path(vault_dir)
    if not root.exists():
        print(f"ERROR: Vault directory {root} not found.")
        return

    # Collect all existing file stems and relative paths
    files = [p for p in root.rglob("*") if p.is_file() and "Datasets" not in p.parts]
    names = {p.name.lower() for p in files} | {
        p.stem.lower() for p in files if p.suffix.lower() == ".md"
    }
    paths = {p.relative_to(root).as_posix().lower() for p in files}
    pat = re.compile(r"\[\[([^\]|#]+)")

    total_links = 0
    resolved_links = 0
    unresolved = []

    # Files to exclude from unresolved reporting (templates and historic reports)
    excluded_dirs = {"Datasets", "Templates", "Laporan", "Arsip"}

    for md in sorted(root.rglob("*.md")):
        if any(d in md.parts for d in excluded_dirs):
            continue

        text = md.read_text(encoding="utf-8", errors="ignore")
        for m in pat.finditer(text):
            total_links += 1
            raw = m.group(1).strip()
            # Normalize path separators
            norm = raw.lower().replace("\\", "/")
            target_stem = norm.split("/")[-1]

            if (
                norm in names
                or norm in paths
                or target_stem in names
                or (norm + ".md") in paths
                or (target_stem + ".md") in names
            ):
                resolved_links += 1
                continue

            unresolved.append((md.relative_to(root).as_posix(), raw))

    print("=== VAULT LINK SCAN REPORT ===")
    print(f"Vault: {root}")
    print(f"Total wikilinks scanned (excl. Templates/Laporan/Arsip): {total_links}")
    print(f"Resolved links: {resolved_links}")
    print(f"Unresolved links: {len(unresolved)}")

    if unresolved:
        print("\n--- UNRESOLVED LINKS ---")
        for src, link in unresolved:
            print(f"  {src} -> [[{link}]]")
    else:
        print("\nALL WIKILINKS IN ACTIVE VAULT RESOLVED.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", default="E:/Projects/thesis")
    args = parser.parse_args()
    scan_vault(args.vault)
