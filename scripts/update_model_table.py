#!/usr/bin/env python3
"""Update the README model availability table from SAELens registry.

Usage:
    poetry run python scripts/update_model_table.py
    poetry run python scripts/update_model_table.py --sae-types res
    poetry run python scripts/update_model_table.py --output catalog.json
"""

import argparse
import sys
from pathlib import Path

# Ensure pipeline is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.discovery import discover_models, save_catalog, generate_readme_table, catalog_summary


def main():
    parser = argparse.ArgumentParser(description="Update model availability table")
    parser.add_argument("--sae-types", help="Filter by SAE type (comma-separated)")
    parser.add_argument("--families", help="Filter by model family (comma-separated)")
    parser.add_argument("--output", help="Save catalog JSON to file")
    parser.add_argument("--readme-section", action="store_true", help="Output README Markdown section")
    parser.add_argument("--probe-s3", action="store_true", help="Probe S3 for batch counts (slow)")
    args = parser.parse_args()

    sae_types = args.sae_types.split(",") if args.sae_types else None
    families = args.families.split(",") if args.families else None

    print("Discovering models from SAELens registry...", file=sys.stderr)
    catalog = discover_models(
        probe_s3=args.probe_s3,
        sae_types=sae_types,
        model_families=families,
    )

    if args.output:
        save_catalog(catalog, Path(args.output))

    if args.readme_section:
        print(generate_readme_table(catalog))
    else:
        print(catalog_summary(catalog))

    print(f"\nTotal: {len(catalog)} SAE configurations", file=sys.stderr)


if __name__ == "__main__":
    main()
