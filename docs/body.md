# Body Analysis and Pose Verification

The `body/` package implements user input validation (FR-1), live camera and static photo
pose estimation (FR-2, FR-7), full-body and front-facing pose verification with user guidance
(FR-8, FR-9), centimetre-per-pixel (`cm_per_pixel`) scale calibration, and anthropometric
measurement extraction producing [`schemas.body.Body`](../schemas/body.py) messages.

## Input Validation (FR-1)

`validate_user_anthropometrics(height_cm, weight_kg)` enforces the project limits before
running calibration or building a `Body` message:

| Parameter | Allowed range | Unit |
|---|---|---|
| `height_cm` | `140.0` to `210.0` | cm |
| `weight_kg` | `40.0` to `150.0` | kg |

Values outside these ranges (or booleans / non-finite floats) raise a `ValueError`.

## Centimetre-per-Pixel Scale Function

`compute_cm_per_pixel(height_cm, full_body_px)` and
`compute_cm_per_pixel_from_landmarks(landmarks, height_cm, frame_height)` calibrate the
camera frame to physical centimetres:

1. **Crown & Sole Estimation (`estimate_body_height_details`)**:
   - Computes the eye-to-heel vertical pixel span and projects the cranial crown using human
     stature proportions (eye-to-crown is $6.4\%$ of standing height, or $0.52 \times$ the
     eye-to-shoulder vertical span).
   - If a foreground segmentation mask is provided, refines the crown and sole coordinates to
     the actual top and bottom foreground pixels in the head/foot column.
2. **Temporal Stabilization (`ScaleCalibrator`)**:
   - Applies a rolling-window median filter with spike dampening across consecutive video
     frames so live centimetre readouts remain steady.

## Five-Rule Pose Verification (`PoseRuleEngine`)

| Rule key | Check | Guidance message when failing |
|---|---|---|
| `visibility` | Nose, eye, both ankles and both heels visible ($\ge 0.60$) inside $2\%$ frame margin | `"Take a step back so head, ankles & heels are visible"` |
| `body_fill` | Body fills $62\%–92\%$ of frame height and shoulder midpoint is centered ($\pm 18\%$) | `"Take a step closer"`, `"Take a step back"`, `"Move to the center of the camera"` |
| `front_facing` | Isotropic shoulder-to-torso pixel ratio ($\ge 0.45$), depth symmetry ($|\Delta z| \le 0.35$), nose centered | `"Face the camera directly"` |
| `arm_spread` | Left and right arm-to-torso angles within $14^\circ–65^\circ$ (A-pose) | `"Spread your arms a little"`, `"Lower your arms slightly"` |
| `shoulders` | Shoulder line tilt $\le 5.0^\circ$ from horizontal | `"Level your shoulders horizontally"` |

When all five rules pass consecutively for `ready_hold_frames` (or when a raised-fist gesture
is detected via `detect_fist_or_hand_trigger` while all rules pass), the frame locks ready for
capture.

## Producing the `Body` Schema Message

```python
from body import BodyAnalysisPipeline

pipeline = BodyAnalysisPipeline(height_cm=175.0, weight_kg=72.0, stabilize_scale=True)
result = pipeline.analyze_landmarks(landmarks, frame_height=720, segmentation_mask=mask)

if result.body_schema is not None:
    print(result.body_schema.model_dump_json(indent=2))
```

## Tape-Measurement Validation (NFR-3)

`evaluate_against_tape_measurements(samples)` compares predicted shoulder width and torso
length against physical tape measurements across participants and checks compliance with NFR-3
(Mean Absolute Error $\le 4.0\text{ cm}$, target $\le 3.0\text{ cm}$).
