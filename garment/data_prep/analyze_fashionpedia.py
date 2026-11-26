"""Phase A2: Fashionpedia annotation counts and analysis.

Input: data/fashionpedia/annotations/instances_attributes_{train,val}2020.json
Outputs (garment/artifacts/analysis/):
    fashionpedia_category_counts.csv
    fashionpedia_attribute_usage.csv
    fashionpedia_axis_coverage.csv
    fashionpedia_top_nickname_distribution.csv
    fashionpedia_axis_value_distribution.csv
Prints a summary. Read-only analysis; no split decisions are made here.
Run: python -m garment.data_prep.analyze_fashionpedia
"""

import csv
import json
from collections import Counter, defaultdict

from .paths import ANALYSIS_OUT, FP_ANN_DIR

# Target upper-body categories (Fashionpedia ids)
CAT_SHIRT = 0        # shirt, blouse
CAT_TOP = 1          # top, t-shirt, sweatshirt
CAT_SWEATER = 2      # sweater (out of scope, tracked for completeness)
CAT_CARDIGAN = 3     # cardigan (out of scope, tracked for completeness)
TARGET_CATEGORIES = {CAT_SHIRT, CAT_TOP, CAT_SWEATER, CAT_CARDIGAN}

# Axis definitions (Fashionpedia attribute id ranges)
AXES = {
    "length": set(range(146, 156)),          # length group: garment length values
    "sleeve": set(range(156, 161)),          # length group: sleeve length values
    "fit": {134, 135, 136, 137, 138},        # silhouette group: fit values
    "neckline": set(range(179, 204)),        # neckline type
    "closure": set(range(225, 235)),         # opening type
}


def load(name):
    with open(FP_ANN_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def main():
    ANALYSIS_OUT.mkdir(parents=True, exist_ok=True)
    val = load("instances_attributes_val2020.json")
    train = load("instances_attributes_train2020.json")

    categories = {c["id"]: c for c in train["categories"]}
    attributes = {a["id"]: a for a in train["attributes"]}

    # Category counts
    mask_count = {"train": Counter(), "val": Counter()}
    image_set = {"train": defaultdict(set), "val": defaultdict(set)}
    for split, d in (("train", train), ("val", val)):
        for a in d["annotations"]:
            cid = a["category_id"]
            mask_count[split][cid] += 1
            image_set[split][cid].add(a["image_id"])

    with open(ANALYSIS_OUT / "fashionpedia_category_counts.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["category_id", "category_name", "supercategory", "masks_train", "masks_val",
                    "images_train", "images_val"])
        for cid in sorted(categories):
            c = categories[cid]
            w.writerow([cid, c["name"], c["supercategory"], mask_count["train"][cid],
                        mask_count["val"][cid], len(image_set["train"][cid]),
                        len(image_set["val"][cid])])

    # Attribute usage (all masks + target upper-body masks)
    attr_total = Counter()
    attr_target = Counter()
    for a in train["annotations"]:
        target = a["category_id"] in TARGET_CATEGORIES
        for aid in a["attribute_ids"]:
            attr_total[aid] += 1
            if target:
                attr_target[aid] += 1

    with open(ANALYSIS_OUT / "fashionpedia_attribute_usage.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["attribute_id", "attribute_name", "group", "usage_train",
                    "usage_upperbody_train"])
        for aid in sorted(attributes):
            a = attributes[aid]
            w.writerow([aid, a["name"], a["supercategory"], attr_total[aid], attr_target[aid]])

    # Axis label coverage on target categories (train+val)
    coverage = {split: {cid: {ax: Counter() for ax in AXES} for cid in sorted(TARGET_CATEGORIES)}
                for split in ("train", "val")}
    axis_value_count = {cid: {ax: Counter() for ax in AXES} for cid in sorted(TARGET_CATEGORIES)}
    top_nickname = Counter()
    top_nickname_hip = Counter()

    for split, d in (("train", train), ("val", val)):
        for a in d["annotations"]:
            cid = a["category_id"]
            if cid not in TARGET_CATEGORIES:
                continue
            aids = set(a["attribute_ids"])
            for ax, id_set in AXES.items():
                found = aids & id_set
                if found:
                    coverage[split][cid][ax]["labeled"] += 1
                    if len(found) > 1:
                        coverage[split][cid][ax]["multi_valued"] += 1
                    if split == "train":
                        for aid in found:
                            axis_value_count[cid][ax][aid] += 1
                else:
                    coverage[split][cid][ax]["unlabeled"] += 1
            if cid == CAT_TOP:
                nick = aids & set(range(0, 18))  # upper-body nicknames
                if 16 in nick:
                    top_nickname["hoodie"] += 1
                elif nick:
                    for aid in nick:
                        top_nickname[attributes[aid]["name"]] += 1
                else:
                    top_nickname["(no nickname)"] += 1
                if 147 in aids:
                    top_nickname_hip["hoodie" if 16 in nick else "other"] += 1

    with open(ANALYSIS_OUT / "fashionpedia_axis_coverage.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["split", "category_id", "category_name", "axis", "labeled_masks",
                    "unlabeled_masks", "multi_valued_masks", "coverage_ratio"])
        for split in ("train", "val"):
            for cid in sorted(TARGET_CATEGORIES):
                for ax in AXES:
                    s = coverage[split][cid][ax]
                    total = s["labeled"] + s["unlabeled"]
                    ratio = (s["labeled"] / total) if total else 0.0
                    w.writerow([split, cid, categories[cid]["name"], ax, s["labeled"],
                                s["unlabeled"], s["multi_valued"], round(ratio, 4)])

    with open(ANALYSIS_OUT / "fashionpedia_top_nickname_distribution.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["nickname", "mask_count_train"])
        for name, n in top_nickname.most_common():
            w.writerow([name, n])

    with open(ANALYSIS_OUT / "fashionpedia_axis_value_distribution.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["category_id", "category_name", "axis", "attribute_id", "attribute_name",
                    "mask_count_train"])
        for cid in sorted(TARGET_CATEGORIES):
            for ax in AXES:
                for aid, n in axis_value_count[cid][ax].most_common():
                    w.writerow([cid, categories[cid]["name"], ax, aid, attributes[aid]["name"], n])

    # Console summary
    print("=== Fashionpedia overview ===")
    print("train images:", len(train["images"]), " train masks:", len(train["annotations"]))
    print("val images:", len(val["images"]), " val masks:", len(val["annotations"]))
    print("\n=== Upper-body categories (train masks / val masks) ===")
    for cid in sorted(TARGET_CATEGORIES):
        print(f"  {cid} {categories[cid]['name']}: "
              f"{mask_count['train'][cid]} / {mask_count['val'][cid]}")
    print("\n=== Axis coverage ratios (train, target categories) ===")
    for cid in sorted(TARGET_CATEGORIES):
        parts = []
        for ax in AXES:
            s = coverage["train"][cid][ax]
            total = s["labeled"] + s["unlabeled"]
            ratio = (s["labeled"] / total * 100) if total else 0.0
            parts.append(f"{ax}={ratio:.1f}%")
        print(f"  {categories[cid]['name']}: " + "  ".join(parts))
    print("\n=== top,t-shirt,sweatshirt nickname distribution (train, top 12) ===")
    for name, n in top_nickname.most_common(12):
        print(f"  {name}: {n}")
    print("\nhip-length top masks, hoodie/other:", dict(top_nickname_hip))
    print("\nCSV outputs:", ANALYSIS_OUT)


if __name__ == "__main__":
    main()
