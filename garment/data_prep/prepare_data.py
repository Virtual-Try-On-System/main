"""Phase C2 + C4: apply the mappings, produce the count report and the fixed splits.

Fashionpedia:
  - Row unit = upper garment MASK (category 0: shirt,blouse / 1: top,t-shirt,sweatshirt)
  - sweatshirt split: nickname=16 (hoodie)
  - length: nickname crop(8) has priority, then length(146-155); fit: silhouette fit(134-138)
  - neckline: from neckline parts (179-203), else collar part nickname(161-173);
    image-level join ONLY on images with exactly one upper-garment mask
  - sleeve: unused (effectively absent in FP) -> empty
  - Split: train2020 images -> train/val (85/15 stratified), official val2020 -> test
Shopping100k:
  - Row unit = image (category 3 Shirt, 4 T-shirt, 5 Jumper)
  - Product group = gender/category/product_id (all views of a product stay in one split)
  - Golden flat pool: 400 images (one per product, class balanced) -> val_flat/test_flat;
    these products are COMPLETELY removed from train/val/test
  - Remaining products: 70/15/15 stratified split
Safety: overlap assertions, file existence checks, SHA256 digests, fixed seed (42).
Run: python -m garment.data_prep.prepare_data
"""

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict

from .analyze_shopping100k import read_attribute
from .paths import ARTIFACTS, FP_ANN_DIR, FP_TRAIN_IMG, FP_VAL_IMG, S100K_IMAGES, SEED

COLUMNS = ["example_id", "source", "domain", "image_path", "category", "product_group",
           "fp_image_id", "fp_annotation_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
           "length", "fit", "sleeve", "neckline", "closure", "label_source"]
AXES = ("length", "fit", "sleeve", "neckline", "closure")


def load_mappings():
    fp_map = {ax: {} for ax in AXES}           # attribute id -> target class (ids unique per axis)
    s_map = defaultdict(dict)                  # (axis, group) -> {value id: target class}
    with open(ARTIFACTS / "mapping_attributes.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            vid = int(r["value_id"])
            if vid < 0:
                continue
            if r["source"] == "fashionpedia":
                fp_map[r["axis"]][vid] = r["target_class"]
            else:
                s_map[(r["axis"], r["group"])][vid] = r["target_class"]
    return fp_map, s_map


def pick_label(aids, mapping):
    """Deterministic smallest-id-first choice; OUT values count as unlabeled."""
    for aid in sorted(aids):
        if aid in mapping:
            target = mapping[aid]
            return "" if target == "OUT" else target
    return ""


def fashionpedia_rows(fp_map):
    result = {"train": [], "test": []}
    missing_files = Counter()
    for split, json_name, img_dir, img_prefix in (
        ("train", "instances_attributes_train2020.json", FP_TRAIN_IMG, "train"),
        ("test", "instances_attributes_val2020.json", FP_VAL_IMG, "val_test2020/test"),
    ):
        with open(FP_ANN_DIR / json_name, encoding="utf-8") as f:
            d = json.load(f)
        file_names = {im["id"]: im["file_name"] for im in d["images"]}

        target_mask_count = Counter()
        neckline_parts = defaultdict(Counter)
        collar_parts = defaultdict(Counter)
        for a in d["annotations"]:
            cid, iid = a["category_id"], a["image_id"]
            if cid in (0, 1):
                target_mask_count[iid] += 1
            elif cid == 33:  # neckline part
                for aid in a["attribute_ids"]:
                    if 179 <= aid <= 203:
                        neckline_parts[iid][aid] += 1
            elif cid == 28:  # collar part
                for aid in a["attribute_ids"]:
                    if 161 <= aid <= 173:
                        collar_parts[iid][aid] += 1

        for a in d["annotations"]:
            cid = a["category_id"]
            if cid not in (0, 1):
                continue
            iid = a["image_id"]
            file_name = file_names[iid]
            if not (img_dir / file_name).is_file():
                missing_files[split] += 1
                continue
            aids = set(a["attribute_ids"])
            category = "shirt" if cid == 0 else ("sweatshirt" if 16 in aids else "tshirt")
            if 8 in aids:
                length = "crop"
            else:
                length = pick_label(aids & set(range(146, 156)), fp_map["length"])
            fit = pick_label(aids & set(range(134, 139)), fp_map["fit"])
            closure = pick_label(aids & set(range(225, 235)), fp_map["closure"])
            neckline = ""
            if target_mask_count[iid] == 1:
                source = neckline_parts[iid] if neckline_parts[iid] else collar_parts[iid]
                if source:
                    aid = min(source, key=lambda x: (-source[x], x))
                    neckline = pick_label({aid}, fp_map["neckline"])
            x, y, w, h = a["bbox"]
            result[split].append({
                "example_id": f"fp_{a['id']}", "source": "fashionpedia", "domain": "on_model",
                "image_path": f"{img_prefix}/{file_name}", "category": category,
                "product_group": f"fpimg_{iid}", "fp_image_id": iid, "fp_annotation_id": a["id"],
                "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
                "length": length, "fit": fit, "sleeve": "", "neckline": neckline,
                "closure": closure, "label_source": "fashionpedia_annotation",
            })
        del d
    print("FP masks skipped (image missing on disk):", dict(missing_files))
    return result


def fashionpedia_split(train_rows, rng):
    groups = defaultdict(list)
    for row in train_rows:
        groups[row["product_group"]].append(row)
    strata = {}
    for group, rows in groups.items():
        classes = {row["category"] for row in rows}
        strata[group] = classes.pop() if len(classes) == 1 else "mixed"
    train_g, val_g = stratified_split(list(groups), strata, (0.85, 0.15), rng)
    train = [row for g in train_g for row in groups[g]]
    val = [row for g in val_g for row in groups[g]]
    return train, val


def shopping100k_rows(s_map):
    category = read_attribute("category.txt")
    fit = read_attribute("fit.txt")
    sleeve = read_attribute("sleevelen.txt")
    collar = read_attribute("collar.txt")
    neckline = read_attribute("neckline.txt")
    fastening = read_attribute("fastening.txt")

    rows = []
    missing = 0
    for path, cid in category.items():
        if cid not in (3, 4, 5):
            continue
        if not (S100K_IMAGES / path).is_file():
            missing += 1
            continue
        label = {3: "shirt", 4: "tshirt", 5: "sweatshirt"}[cid]
        fit_label = ""
        v = fit.get(path, -1)
        if v != -1:
            target = s_map[("fit", "fit")].get(v, "")
            fit_label = "" if target == "OUT" else target
        sleeve_label = ""
        v = sleeve.get(path, -1)
        if v != -1:
            sleeve_label = s_map[("sleeve", "sleevelen")].get(v, "")
        neckline_label = ""
        vc = collar.get(path, -1)
        vn = neckline.get(path, -1)
        if vc != -1:
            neckline_label = s_map[("neckline", "collar")].get(vc, "")
        elif vn != -1:
            neckline_label = s_map[("neckline", "neckline")].get(vn, "")
        closure_label = ""
        v = fastening.get(path, -1)
        if v != -1:
            closure_label = s_map[("closure", "fastening")].get(v, "")
        parts = path.split("/")
        product = parts[-1].rsplit(".", 1)[0].rsplit("_", 1)[0]
        group = "/".join(parts[:-1]) + "/" + product
        rows.append({
            "example_id": "s100k_" + path.replace("/", "_").replace(".jpg", ""),
            "source": "shopping100k", "domain": "flat", "image_path": path, "category": label,
            "product_group": group, "fp_image_id": "", "fp_annotation_id": "",
            "bbox_x": "", "bbox_y": "", "bbox_w": "", "bbox_h": "",
            "length": "", "fit": fit_label, "sleeve": sleeve_label,
            "neckline": neckline_label, "closure": closure_label,
            "label_source": "shopping100k_attributes",
        })
    print("S100k images skipped (missing on disk):", missing)
    return rows


def stratified_split(groups, strata, ratios, rng):
    """Split group keys by stratum with the given ratios (must sum to 1)."""
    assert abs(sum(ratios) - 1.0) < 1e-9
    by_stratum = defaultdict(list)
    for g in sorted(groups):
        by_stratum[strata[g]].append(g)
    result = [[] for _ in ratios]
    for _stratum, keys in by_stratum.items():
        keys = list(keys)
        rng.shuffle(keys)
        n = len(keys)
        k1 = int(round(n * ratios[0]))
        if len(ratios) == 2:
            result[0].extend(keys[:k1])
            result[1].extend(keys[k1:])
        else:
            k2 = int(round(n * ratios[1]))
            result[0].extend(keys[:k1])
            result[1].extend(keys[k1:k1 + k2])
            result[2].extend(keys[k1 + k2:])
    return result


def main():
    rng = random.Random(SEED)
    fp_map, s_map = load_mappings()

    print("--- Fashionpedia rows ---")
    fp = fashionpedia_rows(fp_map)
    print("FP train masks:", len(fp["train"]), " FP test (official val) masks:", len(fp["test"]))
    fp_train, fp_val = fashionpedia_split(fp["train"], rng)
    fp_test = fp["test"]
    print("FP split -> train:", len(fp_train), " val:", len(fp_val), " test:", len(fp_test))

    print("--- Shopping100k rows ---")
    s_rows = shopping100k_rows(s_map)
    groups = defaultdict(list)
    for row in s_rows:
        groups[row["product_group"]].append(row)
    class_groups = defaultdict(list)
    for group, rows in groups.items():
        class_groups[rows[0]["category"]].append(group)

    # Golden flat pool: 134/133/133 products per class (400 total), one image per product
    golden_target = {"tshirt": 134, "shirt": 133, "sweatshirt": 133}
    golden_val, golden_test, remaining_groups = [], [], {}
    for label, group_list in class_groups.items():
        group_list = sorted(group_list)
        rng.shuffle(group_list)
        n_golden = golden_target[label]
        assert len(group_list) > n_golden, f"not enough products for class {label}"
        golden_g = group_list[:n_golden]
        remaining_groups[label] = group_list[n_golden:]
        for i, g in enumerate(golden_g):
            chosen = dict(rng.choice(groups[g]))
            chosen["label_source"] = "shopping100k_prelabel_MANUAL_CORRECTION_REQUIRED"
            # class-balanced halves: even index val, odd index test
            (golden_val if i % 2 == 0 else golden_test).append(chosen)
    print("Golden pool -> val_flat:", len(golden_val), " test_flat:", len(golden_test))

    # Remaining products: 70/15/15
    s_train, s_val, s_test = [], [], []
    for label, group_list in remaining_groups.items():
        strata = {g: label for g in group_list}
        t, v, te = stratified_split(group_list, strata, (0.70, 0.15, 0.15), rng)
        for target, keys in ((s_train, t), (s_val, v), (s_test, te)):
            for g in keys:
                target.extend(groups[g])
    print("S100k split -> train:", len(s_train), " val:", len(s_val), " test:", len(s_test))

    # Safety assertions
    def groups_of(rows):
        return {row["product_group"] for row in rows}

    fp_t, fp_v, fp_te = map(groups_of, (fp_train, fp_val, fp_test))
    assert not (fp_t & fp_v), "FP train/val overlap!"
    assert not (fp_t & fp_te), "FP train/test overlap!"
    assert not (fp_v & fp_te), "FP val/test overlap!"
    s_t = groups_of(s_train)
    s_v = groups_of(s_val)
    s_te = groups_of(s_test)
    golden_g = groups_of(golden_val) | groups_of(golden_test)
    for name, split_set in (("val", s_v), ("test", s_te), ("golden", golden_g)):
        assert not (s_t & split_set), f"S100k train/{name} overlap!"
    assert not (s_v & s_te), "S100k val/test overlap!"
    assert not (s_v & golden_g) and not (s_te & golden_g), "S100k val/test vs golden overlap!"
    assert not (groups_of(golden_val) & groups_of(golden_test)), "val_flat/test_flat overlap!"
    print("Overlap assertions: OK (no group appears in more than one split)")

    def write_csv(name, rows):
        path = ARTIFACTS / name
        rows = sorted(rows, key=lambda r: r["example_id"])
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        return path, rows

    outputs = {
        "train.csv": write_csv("train.csv", fp_train + s_train),
        "val.csv": write_csv("val.csv", fp_val + s_val),
        "test.csv": write_csv("test.csv", fp_test + s_test),
        "val_flat.csv": write_csv("val_flat.csv", golden_val),
        "test_flat.csv": write_csv("test_flat.csv", golden_test),
    }

    with open(ARTIFACTS / "split_sha256.txt", "w", encoding="utf-8") as f:
        for name, (path, rows) in outputs.items():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            f.write(f"{digest}  {name}  ({len(rows)} rows, seed={SEED})\n")
    print("Split files + SHA256 written.")

    # class_counts.csv
    with open(ARTIFACTS / "class_counts.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["source", "split", "axis", "class", "count"])
        splits = {"train": fp_train + s_train, "val": fp_val + s_val, "test": fp_test + s_test,
                  "val_flat": golden_val, "test_flat": golden_test}
        for source in ("fashionpedia", "shopping100k"):
            all_rows = [r for rows in splits.values() for r in rows if r["source"] == source]
            for split_name, rows in list(splits.items()) + [("all", all_rows)]:
                rows = [r for r in rows if r["source"] == source]
                if not rows:
                    continue
                counts = Counter(r["category"] for r in rows)
                for label, n in sorted(counts.items()):
                    w.writerow([source, split_name, "category", label, n])
                for ax in AXES:
                    counts = Counter(r[ax] if r[ax] else "(unlabeled)" for r in rows)
                    for label, n in sorted(counts.items()):
                        if label == "(unlabeled)" and split_name != "all":
                            continue
                        w.writerow([source, split_name, ax, label, n])
    print("class_counts.csv written.")

    # Kappa labeling templates (first 20 val_flat images)
    kappa_rows = sorted(golden_val, key=lambda r: r["example_id"])[:20]
    for person in ("A", "B"):
        with open(ARTIFACTS / f"kappa_labeling_{person}.csv", "w", newline="",
                  encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "length", "fit", "sleeve"])
            for r in kappa_rows:
                w.writerow([r["image_path"], "", "", ""])
    print("kappa_labeling_A/B.csv templates written (20 shared images).")

    print("\n=== ALL axis distributions ===")
    for source in ("fashionpedia", "shopping100k"):
        all_rows = [r for rows in splits.values() for r in rows if r["source"] == source]
        print(f"--- {source} ({len(all_rows)} rows) ---")
        print("  category:", dict(Counter(r["category"] for r in all_rows)))
        for ax in AXES:
            print(f"  {ax}:", dict(Counter(r[ax] if r[ax] else "(unlabeled)" for r in all_rows)))


if __name__ == "__main__":
    main()
