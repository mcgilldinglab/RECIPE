from __future__ import annotations

import argparse
import json

from _bootstrap import add_src_to_path

add_src_to_path()

from recipe.data_construction import build_data_aliases


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RECIPE data aliases from RECIPE_SOURCE_DATA_ROOT.")
    parser.add_argument("--manifest-json", default="data/alias_manifest.json")
    args = parser.parse_args()
    summary = build_data_aliases(output_manifest_json=args.manifest_json)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
