# Dissertation-Ready Summary

## Thesis Claim

This project studies whether a reservoir-computing Ant controller can inherit structured locomotion priors from three friction-specific CPG teachers and then adapt those priors across different floor surfaces. The dissertation-safe claim is that the architecture supports surface-conditioned adaptation and produces more readable, better controlled quadruped motion than earlier drafts of the gait teacher. It is not yet correct to claim that the final closed-loop RC policy reaches the same long-distance target on every surface.

## System Overview

The implemented pipeline is:

1. Build one friction-specific CPG teacher for each surface condition.
2. Collect supervised rollout episodes from those teachers.
3. Train a NumPy autoencoder to compress the Ant observation vector.
4. Train an Echo State Network readout on the encoded trajectories.
5. Fine-tune per friction with DAgger-style closed-loop updates.

The current implementation still uses three separate trained CPGs, one each for friction `0.5`, `1.0`, and `1.5`, and feeds them into a reservoir computer with per-friction readouts. The saved model is:

- `optimized_rc_80_model/rc_80_optimized.pkl`

## What Changed

The main improvement was in the gait teacher rather than the reservoir itself. Earlier versions produced diagonal-leg hanging or one-leg compensation. The teacher was revised into a stance-heavy quadruped walk with friction-specific phase spacing and duty-factor control so the rendered motion looks more animal-like and less brittle.

The visualization layer was also cleaned up so the evidence is easier to defend in a viva or dissertation defense:

- per-surface stats HUD
- explicit surface switching annotations
- colored surface and GUI cues
- side-by-side clips that make motion speed differences visible

## Best Evidence Clips

Use these clips as the main dissertation evidence:

- Motion and pace contrast: `artifacts/videos/four_leg_speed_distinct_final.mp4`
- Colored GUI and surface labeling: `artifacts/videos/four_leg_equal_segments_colored_gui_final_02.mp4`

If you need a backup clip for surface timing, the equal-segment version is still useful:

- `artifacts/videos/four_leg_equal_segments_final.mp4`

## Latest Saved Results

The latest saved model summary reports:

- Friction `0.5`: `RC=-0.108`, `CPG=0.147`
- Friction `1.0`: `RC=1.125`, `CPG=3.565`
- Friction `1.5`: `RC=2.672`, `CPG=6.762`
- Average alignment: `-0.8%`

These results are dissertation-safe if presented as the current closed-loop RC performance, not as a claim that the model has already solved all three surfaces equally well.

## Recommended Wording

Use wording like this in the dissertation:

"A friction-conditioned reservoir-computing controller was trained from three surface-specific CPG teachers. The resulting system improves the readability and surface sensitivity of the Ant gait, but its final closed-loop performance remains surface dependent and does not yet fully match the best teacher behavior on every friction level."

## What Not To Claim

Do not claim that:

- the final RC policy reaches `5 m` on all three surfaces,
- the long-dwell rollout is a final solved result,
- the reservoir alone caused the improvement without the teacher redesign,
- the current evidence supports a blanket claim that every dwell schedule works equally well.

## Files To Cite Internally

- `scripts/train_rc_optimized_80.py`
- `optimized_rc_80_model/summary.txt`
- `dissertation_training_and_data.md`
- `paper_ready_summary.md`
