"""Phase A3: Shopping100k verification and counting.

- Reads the attribute text files (category, fit, sleevelen, collar, neckline, fastening, gender)
- Counts axis value distributions for the target categories (Shirt=3, T-shirt=4, Jumper=5)
- Checks that every listed image exists on disk
- Produces a random 100-image check list and a contact sheet for the flat-lay check

Outputs: garment/artifacts/analysis/shopping100k_axis_value_distribution.csv
         garment/artifacts/analysis/shopping100k_check_100.csv
         outputs/shopping100k_check_100.jpg  (git-ignored, visual inspection only)

The contact sheet needs Pillow; it is imported lazily so that importing this
module from the other data prep scripts works without Pillow installed.
Run: python -m garment.data_prep.analyze_shopping100k
"""

import csv
import random
from collections import Counter, defaultdict

from .paths import ANALYSIS_OUT, OUTPUTS, S100K_ATTR, S100K_IMAGES, SEED

CATEGORY = {0: "Coat", 1: "Jacket", 2: "Suit", 3: "Shirt", 4: "T-shirt", 5: "Jumper",
            6: "Shorts", 7: "Trouser", 8: "Jean", 9: "Swimming", 10: "Jumpsuit", 11: "Pyjamas",
            12: "Tracksuit Bottoms", 13: "Tracksuit", 14: "Skirt", 15: "Dress"}
FIT = {0: "Bootcut", 1: "Flared", 2: "High waist", 3: "Jeggings", 4: "Large", 5: "Loose", 6: "Low",
       7: "Oversize", 8: "Regular", 9: "Skinny", 10: "Slim", 11: "Small", 12: "Straight",
       13: "Tailered", 14: "Tapered"}
SLEEVE = {0: "3/4", 1: "Spaghetti", 2: "Sleeveless", 3: "Elbow", 4: "Extra Long", 5: "Extra Short",
          6: "Long", 7: "Short", 8: "Strapless"}
COLLAR = {0: "Buttondown", 1: "Cutaway", 2: "High", 3: "Hood", 4: "Kent", 5: "Lapel", 6: "Lined",
          7: "Mandarin", 8: "Polo", 9: "Round", 10: "Shawl", 11: "Turndown", 12: "V-neck",
          13: "Peter Pan", 14: "Volant", 15: "Shirt", 16: "Chin"}
NECKLINE = {0: "Boat", 1: "Backless", 2: "Cache-coeur", 3: "Henley", 4: "Low v-neck",
            5: "Low round",
            6: "Off-the-shoulder", 7: "Round", 8: "Square", 9: "V-neck", 10: "Envelope"}
FASTENING = {0: "Zip", 1: "Back closure", 2: "Belt", 3: "Button", 4: "Covered", 5: "Hook",
             6: "Laces", 7: "No closure", 8: "Front", 9: "Others"}
GENDER = {0: "Male", 1: "Female"}
TARGET_CATEGORIES = {3, 4, 5}


def read_attribute(filename):
    """Read an attribute file: '<image path> <int value>' per line (-1 = missing)."""
    result = {}
    with open(S100K_ATTR / filename, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().rsplit(" ", 1)
            if len(parts) == 2:
                result[parts[0]] = int(parts[1])
    return result


def main():
    ANALYSIS_OUT.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    category = read_attribute("category.txt")
    fit = read_attribute("fit.txt")
    sleeve = read_attribute("sleevelen.txt")
    collar = read_attribute("collar.txt")
    neckline = read_attribute("neckline.txt")
    fastening = read_attribute("fastening.txt")
    gender = read_attribute("gender.txt")

    print("category.txt rows:", len(category))
    missing_files = 0
    distribution = {ax: defaultdict(Counter)
                    for ax in ("fit", "sleeve", "collar", "neckline", "fastening")}
    category_counts = Counter()
    gender_counts = Counter()
    for path, cid in category.items():
        if not (S100K_IMAGES / path).is_file():
            missing_files += 1
            continue
        if cid not in TARGET_CATEGORIES:
            continue
        category_counts[cid] += 1
        gender_counts[gender.get(path, -1)] += 1
        for ax, source, names in (("fit", fit, FIT), ("sleeve", sleeve, SLEEVE),
                                  ("collar", collar, COLLAR), ("neckline", neckline, NECKLINE),
                                  ("fastening", fastening, FASTENING)):
            value = source.get(path, -1)
            name = names.get(value, "UNLABELED(-1)") if value != -1 else "UNLABELED(-1)"
            distribution[ax][cid][name] += 1

    print("images missing on disk:", missing_files)
    print("target category image counts:", {CATEGORY[k]: v for k, v in category_counts.items()})
    print("gender in target categories:",
          {GENDER.get(k, "unknown"): v for k, v in gender_counts.items()})

    out_csv = ANALYSIS_OUT / "shopping100k_axis_value_distribution.csv"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["axis", "category_id", "category_name", "source_value", "image_count"])
        for ax in distribution:
            for cid in sorted(distribution[ax]):
                for value, n in distribution[ax][cid].most_common():
                    w.writerow([ax, cid, CATEGORY[cid], value, n])

    for ax in distribution:
        print(f"\n=== {ax} (target categories) ===")
        for cid in sorted(distribution[ax]):
            print(" ", CATEGORY[cid], dict(distribution[ax][cid].most_common()))

    # Random 100-image check list + contact sheet for the flat-lay verification
    rng = random.Random(SEED)
    sample = rng.sample(sorted(category.keys()), 100)
    with open(ANALYSIS_OUT / "shopping100k_check_100.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["index", "image_path", "category", "flat", "person_present", "note"])
        for i, path in enumerate(sample, 1):
            w.writerow([i, path, CATEGORY[category[path]], "", "", ""])

    from PIL import Image, ImageDraw  # lazy: only needed for the contact sheet

    cell, cols = 160, 10
    sheet = Image.new("RGB", (cols * (cell + 4), cols * (cell + 22)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, path in enumerate(sample):
        try:
            im = Image.open(S100K_IMAGES / path).convert("RGB")
            im.thumbnail((cell, cell))
            x = (i % cols) * (cell + 4)
            y = (i // cols) * (cell + 22)
            sheet.paste(im, (x, y + 18))
            draw.text((x + 2, y + 2), f"{i + 1}", fill="black")
        except OSError as exc:
            print("could not open:", path, exc)
    sheet_path = OUTPUTS / "shopping100k_check_100.jpg"
    sheet.save(sheet_path, quality=85)
    print(f"\nCheck list and contact sheet written (100 images, seed={SEED}): {sheet_path}")


if __name__ == "__main__":
    main()
