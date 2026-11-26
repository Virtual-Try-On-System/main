# Cut Taxonomy and Labeling Guide (Task 3, Goal 2)

Version: 1.0 — For labelers in Task 1 and Task 2.
Labels apply to **flat-lay** garment images. Goal: two people looking at the same image assign the same label.

## 1. Category (required, single class)

| Class | Visual definition | Included | Excluded |
|---|---|---|---|
| `tshirt` | Light/medium upper garment; short, long or sleeveless | t-shirt, polo t-shirt, tank top, undershirt, camisole | hooded thick tops, sweaters |
| `sweatshirt` | Thick top, usually with ribbed cuffs; hooded (hoodie) or not | hoodie, sweatshirt | sweater, cardigan (OUT) |
| `shirt` | Buttoned, collared woven top | shirt, blouse | polo t-shirt (tshirt even with a polo collar) |

Boundary cases:
- **Any hood** → category is always `sweatshirt` (never tshirt).
- **Polo collar + 2-3 button placket + knit fabric** → `tshirt`.
- **Sweater/cardigan** → outside the three classes; do not label, mark `OUT`.
- **Multiple garments/sets** in one image or garment not clearly visible → mark `UNCLEAR`; the image is removed from the pool.

## 2. Length axis (ordinal): crop < short < regular < long

Judge the **vertical garment length**, from shoulder to hem; ignore sleeves.

| Class | Visual definition | Boundary case |
|---|---|---|
| `crop` | Above the waist; midriff exposed | Ends at the waistline -> `short`, not crop |
| `short` | Above the hip | Waist-to-hip indecision defaults to `short` |
| `regular` | Hip length; standard t-shirt/shirt | Just covers the hip -> `regular` |
| `long` | Below the hip and longer (tunic, knee, midi, maxi) | Extends more than 2-3 cm past the hip -> `long` |

Note: Shopping100k has **no length attribute**; this axis is learned only from manual labels on flat-lay images.

## 3. Fit axis (ordinal): tight < regular < loose < oversize

How closely the garment fits. On flat-lay images judge the **chest/waist width relative to length** and the cut.

| Class | Visual definition | Boundary case |
|---|---|---|
| `tight` | Body-hugging; skinny, slim, small | Curved waistline and pronounced chest/waist difference -> `tight` |
| `regular` | Standard cut; regular, tailored | Most common indecision is tight vs regular; straight side seams -> `regular` |
| `loose` | Loose but not oversized; large, loose | Shoulder seam at the natural shoulder line -> `loose` |
| `oversize` | Clearly large/baggy; dropped shoulders, wide body | Shoulder seam drops below the shoulder -> `oversize` |

**Order matters:** predicting `regular` for `tight` is a mild error; predicting `oversize` for `tight` is a severe one.
When unsure, pick the **neighbouring class**, never jump to an extreme.

## 4. Sleeve axis (ordinal): sleeveless < short < three_quarter < long

| Class | Visual definition | Boundary case |
|---|---|---|
| `sleeveless` | No sleeves; spaghetti straps and strapless included | A 1-2 cm fabric extension at the shoulder is not a sleeve |
| `short` | Upper arm; **elbow-length included** | At the elbow -> `short`; 2-3 cm below the elbow -> `three_quarter` |
| `three_quarter` | Below the elbow, above the wrist (3/4) | Near the wrist but not at it -> `three_quarter` |
| `long` | Wrist length and longer (extra long included) | At or past the wrist -> `long` |

If the two sleeves look different (one folded/rolled): use the **longer** one.
Cuffs do not change the class; a rolled-up long sleeve still counts as `long`.

## 5. Neckline axis (nominal, auxiliary): round, v_neck, polo, shirt_collar, hooded, mandarin, boat, square, off_shoulder, collarless, other

- This axis is **not ordinal**; there is no neighbouring-class concept.
- A hood means `hooded`; a shirt collar with a button placket means `shirt_collar`.
- Retailer labels are noisy (e.g. Shopping100k `Square` is over-represented). Pick the closest visual definition; if unsure, add a note instead of forcing `other`.

## 6. Closure axis (nominal, auxiliary): button, zip, none, other

No front opening -> `none`. If both buttons and a zip are present, pick the dominant one and add a note.

## 7. General rules

1. **No label = leave blank.** If an axis cannot be judged, leave the cell empty (not `-1`); never force a class.
2. The guide and thresholds are never tuned by looking at the test split; guide changes use validation examples only.
3. Every labeling session uses the same guide version; after a version change, 20 previously labeled examples are re-checked.

## 8. Labeler agreement (Cohen's kappa) procedure

1. **20 shared images** from the flat golden pool are labeled independently by two labelers
   (files: `garment/artifacts/kappa_labeling_A.csv` and `kappa_labeling_B.csv`; columns: `image_id,length,fit,sleeve`).
2. Run `python garment/data_prep/cohen_kappa.py`.
3. Ordinal axes (length, fit, sleeve) use **weighted kappa**; nominal axes use plain kappa.
4. Target: **at least 0.6** per axis. Below target, the guide's boundary cases are revised and the
   measurement is repeated on 20 new shared images. Results go to `garment/taxonomy/kappa_report.md`.
