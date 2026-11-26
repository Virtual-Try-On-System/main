# Body Analysis & Pose Verification Module (`body/`)

This module implements real-time camera capture, MediaPipe pose estimation, the five-rule
full-body & front-facing pose verification engine, the centimetre-per-pixel (`cm_per_pixel`)
scale calibration system, and anthropometric body measurement extraction producing validated
[`schemas.body.Body`](../schemas/body.py) messages.

## Overview of Components

| File | Responsibility |
|---|---|
| `body/camera.py` | OpenCV camera stream (`CameraStream`) and rolling-window frame rate counter (`FPSCounter`, NFR-1). |
| `body/landmarks.py` | MediaPipe 33-point pose landmark enumeration (`PoseLandmarkIndex`), skeleton topology (`POSE_CONNECTIONS`), and frame-boundary check (`LandmarkPoint`). |
| `body/pose_detector.py` | MediaPipe Pose wrapper (`PoseDetector`) supporting both live video (`RunningMode.VIDEO`) and uploaded static photos (`RunningMode.IMAGE`, FR-2), plus skeleton and segmentation contour rendering. |
| `body/pose_rules.py` | Five-rule full-body verification engine (`PoseRuleEngine`), guidance message smoother, countdown lock, and raised-hand/fist gesture detector (`detect_fist_or_hand_trigger`, FR-8, FR-9). |
| `body/scale.py` | Centimetre-per-pixel scale calibration (`compute_cm_per_pixel`, `estimate_body_height_details`, `ScaleCalibrator`) with proportional crown estimation, segmentation mask crown/sole refinement, and temporal outlier-rejected median smoothing. |
| `body/anthropometry.py` | Pixel-to-centimetre body measurement extractor (`AnthropometricExtractor`), weight-adjusted torso width estimation, `schemas.body.Body` builder (`to_body_schema`), and NFR-3 tape-measurement error evaluator (`evaluate_against_tape_measurements`). |
| `body/pipeline.py` | High-level coordinator (`BodyAnalysisPipeline`) combining pose rules, scale calibration, and `Body` schema generation in a single call. |
| `body/ui_overlay.py` | OpenCV HUD overlay (`UIOverlay`) rendering live FPS badge, guidance banner, countdown progress bar, 5-item pose checklist, and anthropometric measurements panel. |

## Pose Verification Rules (FR-7, FR-9)

Every frame is checked against five ordered rules matching the Capture screen specification:

1. **Head, Ankle & Heel Visibility (`visibility`)**: Nose, at least one eye, both ankles, and both heels must have visibility $\ge 0.60$ and lie inside the $2\%$ frame margin.
   - Guidance: `"Take a step back so head, ankles & heels are visible"`
2. **Body Frame Fill & Centering (`body_fill`)**: Full-body vertical span must fill $62\%–92\%$ of the frame height, and shoulder midpoint must be within $\pm 18\%$ of horizontal center.
   - Guidance: `"Take a step closer"`, `"Take a step back"`, or `"Move to the center of the camera"`
3. **Facing the Camera (`front_facing`)**: Verifies frontal orientation in isotropic pixel coordinates using shoulder-to-torso ratio ($\ge 0.45$), left/right shoulder depth symmetry ($|\Delta z| \le 0.35$), and nose centering between shoulders.
   - Guidance: `"Face the camera directly"`
4. **Arm Spread / A-Pose (`arm_spread`)**: Left and right arm-to-torso abduction angles must both lie between $14^\circ$ and $65^\circ$.
   - Guidance: `"Spread your arms a little"` or `"Lower your arms slightly"`
5. **Horizontal Shoulder Line (`shoulders`)**: Shoulder line tilt angle must not exceed $5.0^\circ$.
   - Guidance: `"Level your shoulders horizontally"`

## Centimetre-per-Pixel Scale Calibration (`body/scale.py`)

The scale factor converts pixel distances to physical centimetres using the user's entered
height $H_{\text{cm}} \in [140, 210]$ (FR-1):

$$\text{cm\_per\_pixel} = \frac{H_{\text{cm}}}{H_{\text{px}}}$$

How $H_{\text{px}}$ (full-body pixel height from crown to sole) is determined:
1. **Proportional Anthropometric Crown Offset**: MediaPipe landmarks provide eye and nose coordinates rather than the top of the cranial crown. Instead of a fixed frame-height percentage (which biases scale when the subject stands closer or farther), `estimate_body_height_details()` derives the eye-to-crown offset from the subject's apparent eye-to-shoulder and eye-to-heel pixel spans ($6.4\%$ of total stature above the eye line).
2. **Segmentation Mask Refinement**: When a binary segmentation mask is available, the vertical band above the head and below the heels is scanned to lock onto the exact top-most crown pixel and bottom-most sole pixel.
3. **Temporal Stabilization (`ScaleCalibrator`)**: In live camera streams, `ScaleCalibrator` maintains a 15-frame rolling median and dampens single-frame jumps $>20\%$.

## CLI Usage

Live webcam preview:

```bash
uv run python main.py --user-height 178 --user-weight 74
```

Static photo analysis (FR-2):

```bash
uv run python main.py --image path/to/person.jpg --user-height 178 --user-weight 74
```
