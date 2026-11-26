"""Phase C1: generate mapping_categories.csv and mapping_attributes.csv.

Category/attribute names are read from the real annotation metadata (no hand-typed names).
These CSVs are the single source of truth; prepare_data.py applies the mappings from them.
Run: python -m garment.data_prep.generate_mappings
"""

import csv
import json

from .analyze_shopping100k import CATEGORY as S_CAT
from .analyze_shopping100k import COLLAR, FASTENING, FIT, NECKLINE, SLEEVE
from .paths import ARTIFACTS, FP_ANN_DIR

# Fashionpedia axis mappings: {attribute_id: (target_class, note)}
FP_LENGTH = {
    146: ("short", "above-the-hip"), 147: ("regular", "hip"),
    148: ("crop", "micro; crop on upper body"), 149: ("crop", "mini; crop on upper body"),
    150: ("long", "above-the-knee"), 151: ("long", "knee"), 152: ("long", "below the knee"),
    153: ("long", "midi"), 154: ("long", "maxi"), 155: ("long", "floor"),
}
FP_LENGTH_NICK = {8: ("crop", "crop (top); PRIORITY rule, overrides length value")}
FP_SLEEVE = {
    156: ("sleeveless", "sleeveless"), 157: ("short", "short"), 158: ("short", "elbow-length"),
    159: ("three_quarter", "three quarter"), 160: ("long", "wrist-length"),
}
FP_FIT = {
    134: ("tight", "curved (fit)"), 135: ("tight", "tight (fit)"),
    136: ("regular", "regular (fit)"), 137: ("loose", "loose (fit)"),
    138: ("oversize", "oversized"),
}
FP_NECKLINE = {
    179: ("collarless", ""), 180: ("other", ""), 181: ("round", ""), 182: ("round", ""),
    183: ("v_neck", ""), 184: ("other", ""), 185: ("round", ""), 186: ("round", ""),
    187: ("other", ""), 188: ("other", ""), 189: ("boat", ""), 190: ("round", ""),
    191: ("square", ""), 192: ("v_neck", "plunging = deep v"), 193: ("other", ""),
    194: ("other", ""), 195: ("other", ""), 196: ("other", ""),
    197: ("mandarin", "high neck = stand collar"), 198: ("mandarin", "turtle = stand collar group"),
    199: ("other", ""), 200: ("other", ""), 201: ("other", ""),
    202: ("off_shoulder", ""), 203: ("off_shoulder", ""),
}
FP_COLLAR_NICK = {
    161: ("other", ""), 162: ("shirt_collar", ""), 163: ("shirt_collar", ""), 164: ("polo", ""),
    165: ("other", ""), 166: ("mandarin", ""), 167: ("mandarin", ""), 168: ("other", ""),
    169: ("other", ""), 170: ("other", ""), 171: ("other", ""), 172: ("other", ""),
    173: ("other", ""),
}
FP_CLOSURE = {
    225: ("button", ""), 226: ("button", ""), 227: ("other", ""), 228: ("other", ""),
    229: ("zip", ""), 230: ("other", ""), 231: ("other", ""), 232: ("other", ""),
    233: ("other", ""), 234: ("none", ""),
}

# Shopping100k axis mappings (trouser-specific fit values are OUT)
S_FIT = {0: "OUT", 1: "OUT", 2: "OUT", 3: "OUT", 4: "loose", 5: "loose", 6: "OUT",
         7: "oversize", 8: "regular", 9: "tight", 10: "tight", 11: "tight", 12: "OUT",
         13: "regular", 14: "OUT"}
S_SLEEVE = {0: "three_quarter", 1: "sleeveless", 2: "sleeveless", 3: "short", 4: "long",
            5: "short", 6: "long", 7: "short", 8: "sleeveless"}
S_COLLAR = {0: "shirt_collar", 1: "shirt_collar", 2: "mandarin", 3: "hooded", 4: "shirt_collar",
            5: "other", 6: "other", 7: "mandarin", 8: "polo", 9: "round", 10: "other",
            11: "shirt_collar", 12: "v_neck", 13: "other", 14: "other", 15: "shirt_collar",
            16: "other"}
S_NECKLINE = {0: "boat", 1: "other", 2: "other", 3: "other", 4: "v_neck", 5: "round",
              6: "off_shoulder", 7: "round", 8: "square", 9: "v_neck", 10: "other"}
S_FASTENING = {0: "zip", 1: "other", 2: "other", 3: "button", 4: "other", 5: "other",
               6: "other", 7: "none", 8: "other", 9: "other"}


def main():
    with open(FP_ANN_DIR / "instances_attributes_val2020.json", encoding="utf-8") as f:
        meta = json.load(f)
    fp_cat = {c["id"]: c["name"] for c in meta["categories"]}
    fp_attr = {a["id"]: a for a in meta["attributes"]}

    with open(ARTIFACTS / "mapping_categories.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["source", "category_id", "category_name", "target_class", "condition", "note"])
        for cid in sorted(fp_cat):
            if cid == 0:
                w.writerow(["fashionpedia", cid, fp_cat[cid], "shirt", "", ""])
            elif cid == 1:
                w.writerow(["fashionpedia", cid, fp_cat[cid], "sweatshirt", "nickname=16 (hoodie)",
                            "hooded top = sweatshirt"])
                w.writerow(["fashionpedia", cid, fp_cat[cid], "tshirt",
                            "nickname!=16 or no nickname", "t-shirt, tank, camisole etc."])
            elif cid in (2, 3):
                w.writerow(["fashionpedia", cid, fp_cat[cid], "OUT", "",
                            "sweater/cardigan out of scope"])
            else:
                w.writerow(["fashionpedia", cid, fp_cat[cid], "OUT", "", ""])
        for cid in sorted(S_CAT):
            if cid == 3:
                w.writerow(["shopping100k", cid, S_CAT[cid], "shirt", "", ""])
            elif cid == 4:
                w.writerow(["shopping100k", cid, S_CAT[cid], "tshirt", "", ""])
            elif cid == 5:
                w.writerow(["shopping100k", cid, S_CAT[cid], "sweatshirt", "",
                            "Jumper also contains knitwear; label is noisy"])
            else:
                w.writerow(["shopping100k", cid, S_CAT[cid], "OUT", "", ""])

    with open(ARTIFACTS / "mapping_attributes.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["source", "axis", "group", "value_id", "value_name", "target_class", "note"])

        def fp_rows(axis, ids_map, group_note=""):
            for aid, (target, note) in ids_map.items():
                a = fp_attr[aid]
                w.writerow(["fashionpedia", axis, a["supercategory"], aid, a["name"], target,
                            (group_note + " " + note).strip()])

        fp_rows("length", FP_LENGTH)
        fp_rows("length", FP_LENGTH_NICK)
        fp_rows("sleeve", FP_SLEEVE,
                "Sleeve labels effectively absent in FP (~0.1%); unused in training.")
        for aid, a in sorted(fp_attr.items()):
            if a["supercategory"] == "silhouette":
                if aid in FP_FIT:
                    target, note = FP_FIT[aid]
                    w.writerow(["fashionpedia", "fit", "silhouette", aid, a["name"], target, note])
                else:
                    w.writerow(["fashionpedia", "fit", "silhouette", aid, a["name"], "OUT",
                                "silhouette shape, not a fit value"])
        fp_rows("neckline", FP_NECKLINE)
        fp_rows("neckline", FP_COLLAR_NICK)
        for aid in range(174, 179):  # lapel nicknames
            a = fp_attr[aid]
            w.writerow(["fashionpedia", "neckline", a["supercategory"], aid, a["name"], "OUT",
                        "lapel type, not a neckline class"])
        fp_rows("closure", FP_CLOSURE)

        for vid, name in sorted(FIT.items()):
            w.writerow(["shopping100k", "fit", "fit", vid, name, S_FIT[vid],
                        "trouser-specific value" if S_FIT[vid] == "OUT" else ""])
        for vid, name in sorted(SLEEVE.items()):
            w.writerow(["shopping100k", "sleeve", "sleevelen", vid, name, S_SLEEVE[vid], ""])
        for vid, name in sorted(COLLAR.items()):
            w.writerow(["shopping100k", "neckline", "collar", vid, name, S_COLLAR[vid],
                        "collar takes priority"])
        for vid, name in sorted(NECKLINE.items()):
            w.writerow(["shopping100k", "neckline", "neckline", vid, name, S_NECKLINE[vid],
                        "used when collar is missing"])
        for vid, name in sorted(FASTENING.items()):
            w.writerow(["shopping100k", "closure", "fastening", vid, name, S_FASTENING[vid], ""])
        w.writerow(["shopping100k", "length", "length", -1, "(attribute does not exist)", "OUT",
                    "Shopping100k has NO length attribute; length axis relies on manual labels"])

    print("mapping_categories.csv and mapping_attributes.csv written to", ARTIFACTS)


if __name__ == "__main__":
    main()
