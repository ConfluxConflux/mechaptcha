# mechaptcha

## Claude Summary of Project

**Core Question:** Is behavioral invariance the same as representational erasure? If you train a CNN to correctly read CAPTCHA text *regardless* of whether a distortion is present, do the model's internal activations still secretly encode that distortion?

**Setup:**
- Generate two synthetic CAPTCHA datasets (5-letter sequences, varied fonts): **Batch A** has a line running through the text, **Batch B** does not — otherwise identical.
- Use an open-source CNN trained on CAPTCHAs to transcribe the text. Training loss encourages the same output for both batches, so the model is behaviorally incentivized to ignore the line.

**Method (mechanistic interpretability):**
- Hook into the CNN at multiple intermediate layers (after each conv, pool, and FC block).
- At each hook point, train a **linear probe** to classify Batch A vs. Batch B from the activations.
- Track how probe accuracy evolves across layers — does the model gradually "forget" the line, or does it retain that information all the way through?

**Hypothesis:** Even in later layers, distortion information remains linearly decodable — behavioral invariance does not imply the representation has erased the irrelevant feature.

**Baselines:** (1) Check if the line actually hurts transcription accuracy (if so, the model detects it trivially); (2) random-label control to confirm probes are finding real signal.