"""Supplementary analysis: sleeve/neckline labels live on part masks.

sleeve=30, neckline=33, collar=28. Measures the image-level join potential on
images with exactly one target upper-garment mask.
Outputs: garment/artifacts/analysis/fashionpedia_part_coverage.csv
         garment/artifacts/analysis/fashionpedia_sleeve_neckline_value_distribution.csv
Run: python -m garment.data_prep.analyze_fashionpedia_parts
"""

import csv
import json
from collections import Counter, defaultdict

from .paths import ANALYSIS_OUT, FP_ANN_DIR

CAT_SLEEVE, CAT_COLLAR, CAT_NECKLINE = 30, 28, 33
TARGET_GARMENTS = {0, 1}  # shirt,blouse and top,t-shirt,sweatshirt
SLEEVE_IDS = set(range(156, 161))
NECKLINE_IDS = set(range(179, 204))
COLLAR_NICK = set(range(161, 174))


def main():
    with open(FP_ANN_DIR / "instances_attributes_train2020.json", encoding="utf-8") as f:
        train = json.load(f)
    attributes = {a["id"]: a for a in train["attributes"]}

    part_count = Counter()
    sleeve_values = Counter()
    neckline_values = Counter()
    collar_values = Counter()
    garment_mask_count = Counter()
    sleeve_by_image = defaultdict(Counter)
    neckline_by_image = defaultdict(Counter)

    for a in train["annotations"]:
        cid = a["category_id"]
        iid = a["image_id"]
        aids = set(a["attribute_ids"])
        if cid in TARGET_GARMENTS:
            garment_mask_count[iid] += 1
        elif cid == CAT_SLEEVE:
            part_count["sleeve"] += 1
            for aid in aids & SLEEVE_IDS:
                sleeve_values[aid] += 1
                sleeve_by_image[iid][aid] += 1
        elif cid == CAT_NECKLINE:
            part_count["neckline"] += 1
            for aid in aids & NECKLINE_IDS:
                neckline_values[aid] += 1
                neckline_by_image[iid][aid] += 1
        elif cid == CAT_COLLAR:
            part_count["collar"] += 1
            for aid in aids & COLLAR_NICK:
                collar_values[aid] += 1

    single_mask_images = {iid for iid, n in garment_mask_count.items() if n == 1}
    sleeve_labeled_images = {iid for iid, c in sleeve_by_image.items() if c}
    neckline_labeled_images = {iid for iid, c in neckline_by_image.items() if c}

    with open(ANALYSIS_OUT / "fashionpedia_part_coverage.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["sleeve_part_masks", part_count["sleeve"]])
        w.writerow(["neckline_part_masks", part_count["neckline"]])
        w.writerow(["collar_part_masks", part_count["collar"]])
        w.writerow(["sleeve_labeled_sleeve_masks", sum(sleeve_values.values())])
        w.writerow(["neckline_labeled_neckline_masks", sum(neckline_values.values())])
        w.writerow(["single_target_garment_images", len(single_mask_images)])
        w.writerow(["single_mask_with_sleeve_label",
                    len(single_mask_images & sleeve_labeled_images)])
        w.writerow(["single_mask_with_neckline_label",
                    len(single_mask_images & neckline_labeled_images)])

    with open(ANALYSIS_OUT / "fashionpedia_sleeve_neckline_value_distribution.csv", "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["axis", "attribute_id", "attribute_name", "part_mask_count_train"])
        for aid, n in sleeve_values.most_common():
            w.writerow(["sleeve", aid, attributes[aid]["name"], n])
        for aid, n in neckline_values.most_common():
            w.writerow(["neckline", aid, attributes[aid]["name"], n])
        for aid, n in collar_values.most_common():
            w.writerow(["collar_nickname", aid, attributes[aid]["name"], n])

    print("part mask counts:", dict(part_count))
    print("sleeve value distribution:",
          {attributes[k]["name"]: v for k, v in sleeve_values.most_common()})
    print("neckline value distribution (top 10):",
          {attributes[k]["name"]: v for k, v in neckline_values.most_common(10)})
    print("single target garment images:", len(single_mask_images))
    print("  with sleeve label (from sleeve parts):",
          len(single_mask_images & sleeve_labeled_images))
    print("  with neckline label (from neckline parts):",
          len(single_mask_images & neckline_labeled_images))


if __name__ == "__main__":
    main()
