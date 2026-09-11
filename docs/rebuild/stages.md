# Delivery stages

## 1. Product flows and browser foundation — implemented, device testing pending

Deliver five connected screens, catalogue, honest historical evidence, camera/clip review, vocabulary lookup, theme preference, local API and reproducible launch commands. Build and API checks pass before handoff. Physical camera behaviour still needs testing on the target Yoga.

## 2. Recovered model integration

Import model metadata/weights with checksums. Restore the exact no-skip baseline and a single shared encoder. Compare known landmark vectors and predictions against the recovered implementation. Connect browser clip processing through MediaPipe with explicit timestamp, hand ordering, padding and model schema. Add loading/error/uncertain/no-hands states. Do not rearrange legacy features without retraining.

Acceptance: one newly captured video produces a traceable result; model, schema and preprocessing are identified; fake/background input does not masquerade as a validated sign. Target hardware latency is measured, not assumed.

## 3. Data workspace

SQLite schema and managed files, provenance-preserving import, fresh recordings, annotations and review status, derivative grouping and saved splits.

Acceptance: a recording can be captured, reviewed, processed and included in one reproducible experiment without losing source relationships. Duplicate imports do not duplicate logical samples.

## 4. Language and visualisation

Restore representative playback, then review text/gloss mapping and speech adapters. Preserve number/repetition semantics; mark unsupported words and missing animations. Expose raw gloss and generated English separately. Collect fluent-reviewer assessments rather than treating visual smoothness as language validity.

Acceptance: both directions have coherent start-to-finish flows within explicitly supported scope, with alternatives for unavailable services.

## 5. Experiments and validation

Background jobs, versioned model registry, grouped evaluation, per-class reports and errors, cancellation, runtime/resource measurements and independent fresh-recording results.

Acceptance: an experiment can be rerun from its manifest. Historical, in-sample, validation and held-out results are distinctly labelled.

## 6. Release and MANAD demonstration

Test installation from a clean checkout plus an explicit asset bundle; review permissions, error recovery, offline operation and accessibility. Produce a versioned release, evidence appendix and NSL demonstration. Explain the project as background for a prospective MSL collaboration, with language/data decisions shaped together.

No deadline or accuracy target is assumed. Complete one stage and verify its meaningful gates before broadening tests or adding architecture complexity.
