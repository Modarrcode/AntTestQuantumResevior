# Results

## Overview

This section reports the final behavior of the Ant reservoir-computing controller after teacher redesign, data collection, autoencoder training, reservoir training, and closed-loop fine-tuning. The results should be interpreted conservatively: the system demonstrates surface-conditioned locomotion and clearer quadruped motion than earlier drafts, but the final closed-loop controller remains surface dependent.

## Final Saved Model Performance

The latest saved model summary reports the following forward distances:

| Friction | RC forward distance | CPG forward distance |
| --- | ---: | ---: |
| 0.5 | -0.108 | 0.147 |
| 1.0 | 1.125 | 3.565 |
| 1.5 | 2.672 | 6.762 |

The average alignment reported by the saved summary is -0.8%.

These numbers show that the final closed-loop RC policy does not yet replicate the best teacher behavior on every surface. Performance is strongest on friction 1.5 among the saved RC runs, but even there the RC remains below the tuned CPG teacher. Friction 0.5 remains the weakest condition and still requires additional tuning if the goal is uniform performance across all surfaces.

## Behavioral Interpretation

The most important result is qualitative as well as quantitative. Earlier gait versions produced unstable diagonal-leg hanging or one-leg compensation. After revising the teacher into a stance-heavy quadruped walk with friction-specific phase spacing and duty-factor control, the rendered motion became more readable and more consistent with four-legged locomotion.

This matters because it isolates the main source of improvement. The reservoir computer provides the learned adaptation layer, but the teacher design determines the quality of the locomotion prior. The results indicate that a better structured teacher produces better evidence for the dissertation, even when the final closed-loop policy is still limited by surface-specific dynamics.

## Evidence To Use In The Dissertation

The most defensible visual evidence is the pair of clips that separate motion pace from GUI labeling:

- `artifacts/videos/four_leg_speed_distinct_final.mp4`
- `artifacts/videos/four_leg_equal_segments_colored_gui_final_02.mp4`

These clips support two distinct points. The first shows that the motion differs meaningfully across surface segments, while the second makes the surface identity explicit through color and GUI cues. Together with the saved model summary, they support the claim that the system is surface-conditioned and visually interpretable, even if it is not yet fully solved.

## Dissertation Conclusion From The Results

The correct conclusion from the current results is not that the model has achieved perfect surface-invariant walking. The correct conclusion is that reservoir computing can absorb and reuse structured CPG priors, and that those priors can be shaped to produce more animal-like quadruped motion across friction conditions. However, the final RC policy still shows uneven performance, so the contribution is best presented as a successful adaptation framework rather than a fully solved locomotion controller.
