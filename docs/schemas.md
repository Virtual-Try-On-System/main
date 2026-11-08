# JSON schemas

Modules exchange three messages, defined with Pydantic in `schemas/`. This is version `1.0` and it
is a **draft until the team locks it**.

| Schema | Produced by | Used by |
|---|---|---|
| `Garment` | `garment/` | `fit/` |
| `Body` | `body/` | `fit/` |
| `FitResult` | `fit/` | `service/`, `evaluation/` |

## Fields

### Garment

| Field | Type | Rule |
|---|---|---|
| `schema_version` | string | Must be `1.0` |
| `category` | enum | `tshirt`, `sweatshirt`, `shirt` |
| `fit_class` | enum | `crop`, `regular`, `oversize` |
| `size_label` | enum | `XS`, `S`, `M`, `L`, `XL`, `XXL` |
| `mask_path` | string | Relative path |
| `ratios.length_to_chest_width` | number | Above 0 |
| `ratios.sleeve_to_length` | number | Above 0 |
| `ratios.shoulder_to_chest_width` | number | Above 0 |

### Body

| Field | Type | Rule |
|---|---|---|
| `schema_version` | string | Must be `1.0` |
| `height_cm` | number | 140 to 210 |
| `weight_kg` | number | 40 to 150 |
| `cm_per_pixel` | number | Above 0 |
| `shoulder_width_cm` | number | Above 0 |
| `torso_length_cm` | number | Above 0 |
| `body_width_cm` | number | Above 0 |
| `pose_keypoints` | list | At least one item: `name`, `x_px`, `y_px`, `confidence` (0 to 1) |

### FitResult

| Field | Type | Rule |
|---|---|---|
| `schema_version` | string | Must be `1.0` |
| `hem_from_shoulder_cm` | number | Above 0 |
| `sleeve_end_from_shoulder_cm` | number | Above 0 |
| `side_width_cm` | number | Above 0 |
| `width_ease_cm` | number | Negative when the garment is tight |
| `length_ease_cm` | number | Negative when the garment is shorter than the reference length |
| `target_mask_path` | string | Relative path |

## General rules

- Unknown fields are rejected, so a typo cannot pass silently.
- Types are strict: `"175"` is not accepted as a number.
- Images are never embedded. Messages hold relative paths only, and `Body` holds no image at all.

## Questions to settle before locking

1. **Ratio vector:** The three ratios are a proposal. Which ones does garment analysis really
   produce?
2. **Size labels:** Is `XS` to `XXL` enough for the size chart draft?
3. **Keypoints:** Do we keep MediaPipe landmark names, and which ones does the fit engine need?
4. **Reference points:** Is the shoulder line the right origin for hem and sleeve positions?
5. **Length ease:** Which body length is it measured against?
6. **Verification:** Should the hem and sleeve error check (FR-10) be part of `FitResult` or a
   separate message?

## Changing a schema

Update the model, its examples and this page in the same pull request, and raise
`schema_version` if the change breaks existing messages.

Invalid examples are named `<field>__<reason>.json`. Each one must break exactly that field, and
the tests check this.
