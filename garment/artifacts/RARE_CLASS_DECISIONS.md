# Rare Class Decisions (Task 3, Goal 3)

Counts come from `class_counts.csv` ("all" rows after the fixed split, seed=42).
Threshold: fewer than ~100 examples per class is considered rare.

## Category

| Class | Fashionpedia | Shopping100k | Decision |
|---|---|---|---|
| tshirt | 16,605 | 22,071 | Sufficient. |
| shirt | 6,263 | 14,853 | Sufficient. |
| sweatshirt | **420** | 11,797 | Only `nickname=hoodie` masks count as sweatshirt in FP; few, but the class is kept (no merge). Main volume is Shopping100k Jumper; **Jumper also contains knitwear, so its labels are noisy** — the two-person manual correction of the golden flat pool measures the real accuracy of this class. |

## Length axis (ordinal)

| Class | Fashionpedia | Shopping100k | Decision |
|---|---|---|---|
| crop | 2,041 | none | Kept (nickname crop-top + micro/mini length). |
| short | 11,811 | none | Sufficient. |
| regular | 5,415 | none | Sufficient. |
| long | **25 (RARE)** | none | **Class kept, but FP alone is insufficient.** Long tops (tunics) are rarely labeled in FP. Mitigation: manual labels of the golden flat pool; if the pool is grown to 1000+ images, long examples will be deliberately oversampled. No merge (long->regular) FOR NOW; revisit with the advisor after the first baseline measurement (Goal 9 note). |
| (unlabeled) | 3,996 | 48,721 | Missing-label loss masking; Shopping100k has no length attribute, so it contributes no loss to the length head. |

## Fit axis (ordinal)

| Class | Fashionpedia | Shopping100k | Decision |
|---|---|---|---|
| tight | 6,570 | 4,721 | Sufficient. |
| regular | 11,283 | 37,353 | Dominant class; class weighting/sampling needed. |
| loose | 1,693 | 5,143 | Sufficient. |
| oversize | **168** | **482** | 650 combined: class kept, no merge. Supported with class weights + augmentation. |

Note: trouser-specific Shopping100k fit values (Bootcut, Flared, High waist, Jeggings, Low,
Straight, Tapered) are OUT; on upper-body images they count as unlabeled.

## Sleeve axis (ordinal)

| Class | Fashionpedia | Shopping100k | Decision |
|---|---|---|---|
| sleeveless | none | 5,218 | Sufficient. |
| short | none | 20,283 | Sufficient (elbow included). |
| three_quarter | none | 2,574 | Sufficient. |
| long | none | 19,892 | Sufficient. |

**Critical:** sleeve-length labels effectively do not exist on Fashionpedia upper-garment
masks (0.1%; 874 sleeve part masks carry zero sleeve labels). FP rows therefore always have
an empty sleeve label; the sleeve head learns from Shopping100k + manual labels only
(missing-label loss masking handles this naturally).

## Neckline axis (nominal, auxiliary)

- FP: small classes such as polo 158, square 171, off_shoulder 262, collarless 436.
  Shopping100k: `square` 23,740 (**over-represented — suspected retailer-label artifact**,
  14,936 of them on T-shirts).
- Decision: neckline stays an **auxiliary axis**; acceptance criteria (Goal 9) apply to the
  main axes (length/fit/sleeve). No merging, but the reliability of the `square` class is
  flagged in the evaluation report. Golden-pool necklines are corrected by two people.

## Closure axis (nominal, auxiliary)

| Class | Fashionpedia | Shopping100k | Decision |
|---|---|---|---|
| button | 5,541 | 7,689 | Sufficient. |
| zip | **66** | 2,399 | Rare in FP; sufficient combined. |
| none | **45 (RARE)** | 0 | `No closure` is never used in the target categories (retailers left -1). Class kept; **garments without a closure (most t-shirts) are UNLABELED, not labeled "none"** (unlabeled != none). "none" examples will be produced by the manual correction of the golden pool. |
| other | 56 | 297 | Auxiliary axis; kept. |

## General decisions

1. No main-axis class was merged or excluded from testing at this stage.
2. For rare classes (length/long, closure/none) the remedy is **manual labeling + class
   weighting**; thresholds and class counts are revisited with the advisor after the first
   baseline measurement (Goals 5/6).
3. sweater/cardigan (FP 2,3) and every other category are OUT — those images never entered
   the train/test rows.
4. The golden flat pool (400 images) is fully excluded from training; its labels are
   Shopping100k pre-labels and **require two-person manual correction** (marked in the
   `label_source` column).
