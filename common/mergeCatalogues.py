"""Merge the three raw Watkins catalogues into data/processed/metadata/mergedCatalogue.txt.

Replaces notebooks/archival/alternate_catalogueMerging.ipynb. This step runs once,
ahead of the whole pipeline, and never per stage - hence a script rather than a
notebook.

    catalogueBCorrection.ipynb  ->  jwst_bubble_properties_B_fixed.txt
    THIS SCRIPT                 ->  mergedCatalogue.txt
    bubbleStrattification.ipynb ->  sizeClassBounds.csv + per-class maps
    [manual cleaning]           ->  badBubbles.txt
    catalogueSegregation.ipynb  ->  cleanedSegregatedMergedCatalogue.txt

GLOBAL_ID IS LOAD-BEARING
    GLOBAL_ID is simply the row number over A, then B_fixed, then C concatenated
    in that order. Nothing else defines it. `badBubbles.txt` - the record of every
    bubble rejected by hand - stores those integers and nothing else, so ANY change
    to the input files, their order, or their row counts silently renumbers the
    whole catalogue and makes the manual cleaning delete the wrong bubbles.

    That is not hypothetical: an obsolete merging notebook once entered the pipeline
    and had to be traced before its output could be trusted.

    So by default this script refuses to overwrite an existing mergedCatalogue.txt
    whose contents would change, and reports how many badBubbles ids still resolve.
    Use --force only when you intend to renumber, and re-do the manual cleaning after.

Run:
    python scripts/mergeCatalogues.py            # write, refusing an unsafe overwrite
    python scripts/mergeCatalogues.py --check    # verify only, write nothing
    python scripts/mergeCatalogues.py --force    # overwrite even if ids would move
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]

# ORDER MATTERS - it defines GLOBAL_ID. Do not sort, do not reorder, do not switch
# to a glob. B is the corrected file from catalogueBCorrection.ipynb, not the raw B.
CATALOGUE_PATHS = {
    "A": REPO / "data/raw/catalogue/jwst_bubble_properties_A.txt",
    "B": REPO / "data/raw/catalogue/jwst_bubble_properties_B_fixed.txt",
    "C": REPO / "data/raw/catalogue/jwst_bubble_properties_C.txt",
}

OUTPUT_PATH = REPO / "data/processed/metadata/mergedCatalogue.txt"
BAD_BUBBLES_PATH = REPO / "data/processed/metadata/badBubbles.txt"

# what the committed catalogue has always contained; a mismatch means an input
# file changed underneath the manual cleaning
EXPECTED_COUNTS = {"A": 1694, "B": 837, "C": 787}


def merge():
    """Concatenate the three catalogues and stamp GLOBAL_ID as row order."""
    all_catalogues = []
    counts = {}

    for catalogue_label, path in CATALOGUE_PATHS.items():
        if not path.exists():
            raise SystemExit(
                f"missing input: {path}\n"
                + ("Run notebooks/catalogueBCorrection.ipynb first - it writes "
                   "jwst_bubble_properties_B_fixed.txt."
                   if catalogue_label == "B" else "")
            )
        df = pd.read_csv(path)
        df["CATALOGUE"] = catalogue_label
        counts[catalogue_label] = len(df)
        all_catalogues.append(df)

    df_all = pd.concat(all_catalogues, ignore_index=True)
    df_all.insert(0, "GLOBAL_ID", range(1, len(df_all) + 1))

    # The notebook also built a `df_reduced` with ARM / DIST_ARM_PC / GAL_RAD_KPC
    # dropped, then saved df_all anyway - the reduced frame was dead code. The drop
    # now happens downstream in catalogueSegregation.ipynb, so the merged file keeps
    # every column and stays a faithful record of the sources.
    return df_all, counts


def report_bad_bubble_impact(df_all):
    """How many manually rejected ids still land inside the new catalogue."""
    if not BAD_BUBBLES_PATH.exists():
        print("  badBubbles.txt not present - no manual cleaning to invalidate")
        return

    bad = pd.read_csv(BAD_BUBBLES_PATH)
    bad.columns = bad.columns.str.strip()
    if "GLOBAL_ID" not in bad.columns:
        print(f"  badBubbles.txt has columns {list(bad.columns)}; expected GLOBAL_ID")
        return

    ids = bad["GLOBAL_ID"].astype(int).drop_duplicates()
    resolved = ids.isin(df_all["GLOBAL_ID"]).sum()
    print(f"  badBubbles.txt: {len(ids)} unique ids, {resolved} resolve, "
          f"{len(ids) - resolved} out of range")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="verify against the existing file and write nothing")
    parser.add_argument("--force", action="store_true",
                        help="overwrite even when GLOBAL_IDs would change")
    args = parser.parse_args()

    df_all, counts = merge()

    print(f"{'catalogue':<12}{'rows':>8}{'expected':>10}   GLOBAL_ID range")
    start = 1
    for label, n in counts.items():
        exp = EXPECTED_COUNTS.get(label)
        flag = "" if exp is None or exp == n else "  <- CHANGED"
        print(f"{label:<12}{n:>8}{exp if exp else '-':>10}   {start}-{start + n - 1}{flag}")
        start += n
    print(f"{'total':<12}{len(df_all):>8}{sum(EXPECTED_COUNTS.values()):>10}")

    drifted = [l for l, n in counts.items()
               if EXPECTED_COUNTS.get(l) not in (None, n)]
    if drifted:
        print(f"\nWARNING: row counts changed for {', '.join(drifted)}. Every GLOBAL_ID "
              "at or after the first changed catalogue has shifted, so badBubbles.txt "
              "no longer refers to the bubbles you rejected.")

    print()
    report_bad_bubble_impact(df_all)

    # compare against what is already on disk
    identical = None
    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)
        identical = existing.equals(df_all)
        print(f"\nexisting {OUTPUT_PATH.name}: {len(existing)} rows, "
              f"{'IDENTICAL to this run' if identical else 'DIFFERS from this run'}")
        if not identical and len(existing) == len(df_all):
            moved = int((existing["GLOBAL_ID"].values != df_all["GLOBAL_ID"].values).sum())
            same_ids = existing["GLOBAL_ID"].equals(df_all["GLOBAL_ID"])
            print(f"  GLOBAL_ID column {'unchanged' if same_ids else f'differs in {moved} rows'}")

    if args.check:
        print("\n--check: nothing written")
        return 0 if identical is not False else 1

    if identical is False and not args.force:
        raise SystemExit(
            "\nREFUSING to overwrite: the merged catalogue would change.\n"
            "badBubbles.txt stores GLOBAL_IDs, so renumbering invalidates the manual\n"
            "cleaning and every downstream file built from it.\n\n"
            "If the change is intended, re-run with --force and then redo the manual\n"
            "cleaning against the new ids."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved full collective catalogue to {OUTPUT_PATH}")
    print(f"  {len(df_all)} bubbles, columns: {list(df_all.columns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
