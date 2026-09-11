# Product and workflows

## Product definition

Sign Avatar is a local-first browser workspace for NSL. The initial baseline is the recovered 49-class recognition family: 26 letters, 10 digits and 13 words. The archive branch preserves the original source and history. The rebuild remains additive while legacy code is progressively replaced with tested services.

Primary users are the demonstrator/researcher and a person trying a supported sign. Data preparation and experiments are researcher workflows. The complete product includes both recognition and sign output; stage 1 does not remove the reverse direction from scope.

## Navigation and screen contracts

| Screen | Primary task | Essential states |
| --- | --- | --- |
| Translate: sign to text | Capture/import one sign, inspect recognition, correct output, optionally synthesise speech | Permission needed/denied, opening, preview, recording, clip ready, processing, no hands, uncertain, result, failure |
| Translate: text to sign | Enter text or transcribed speech, inspect gloss/coverage, review and play signs | Empty, processing, supported/unsupported content, review needed, playable sequence, missing asset, error |
| Sign library | Find a label and inspect its meaning, source, variants and examples | Search results, no matches, selected detail, review status, unavailable example |
| Dataset | Record/import, annotate, review and organise original/derived samples | Empty, importing, invalid file, duplicate, pending review, accepted/rejected, processing failure |
| Experiments | Select data/model configuration, run evaluation/training, inspect reproducible results | Prerequisites missing, configured, queued, running, completed, failed, cancelled |
| Settings | Manage devices, storage, language resources, model choice and appearance | Loading, valid, unsaved, saving, saved, invalid/unavailable |

## Recognition flow

1. Choose camera capture or a local video. Request permission only after an explicit action.
2. Preview hands/body positioning. Keep recording start/end under user control.
3. Record an isolated sign, stop, review, retake or select a different clip.
4. Process with a versioned feature encoder and the matching model checkpoint.
5. Present the raw gloss, model score and useful alternatives. Missing detection and uncertainty must remain explicit.
6. Let a user correct a result. A correction is an annotation, not immediate model retraining.
7. Offer text/speech output when the corresponding capability is installed. Store recordings only through an explicit dataset action.

Stage 1 implements steps 1–3 and download; subsequent inference is visibly unavailable.

## Text/speech to sign flow

1. Enter written English or request speech capture when implemented.
2. Show the transcript for correction before conversion.
3. Convert to gloss with unsupported content, ambiguity and coverage visible. Preserve repeated signs and numerical digits.
4. Review the proposed sequence; never silently omit unsupported meanings.
5. Play the approved sign representations with pause, repeat and speed controls.
6. Keep linguistic review separate from successful software playback.

Stage 1 supports exact label lookup only. Letters are identified as alphabet lookups. English pronouns must use a distinct semantic mapping in the later language stage.

## Dataset flow and planned data model

Recording -> source sample -> annotation -> quality/linguistic review -> derived landmarks -> grouped split -> experiment.

Planned records: Sign, SignVariant, SourceRecording, Signer, RecordingSession, Annotation, DerivedSample, SplitManifest, ModelArtifact, ExperimentRun and ReviewDecision. Stable sign IDs must not depend on display wording. Original recordings and flipped/augmented versions share an origin ID. Signer/session identity is separate from public display metadata.

The recovered 4,900 landmark files include 2,450 original-named and 2,450 flipped samples. They are not 4,900 independent recordings. Original video files were not found in reachable history. Import must retain original IDs, hashes, provenance and uncertainty about unavailable source media.

SQLite will hold records; managed directories will hold videos, arrays and checkpoints. It is not yet implemented. Future imports require type/size validation and paths confined to managed storage. Ordinary metadata should use JSON; any legacy pickle import must be explicit and restricted to trusted recovery assets.

## Validation flow

Record model/code/data versions -> choose grouping -> freeze split manifest -> run evaluation -> save predictions and per-class report -> review errors -> select next experiment.

Historical checkpoint validation scores were used for model selection. They are not independent tests. Historical filename order was not preserved, so random_state alone cannot reconstruct the exact old split. New train/validation/test groups must keep original/derived samples together and preferably hold out signers; with limited people, hold out sessions and label that restriction.

Report sample support, macro F1, confusion matrices, rejection/coverage, errors on unknown/background input and acquisition-to-result latency. Do not report accepted-prediction accuracy alone. Continuous signing requires a separate corpus, segmentation and sequence evaluation.

## UI direction

A quiet working surface: deep green navigation, crisp neutral panels, strong headings and a large signing viewport. Actions remain close to input/output. No promotional landing page precedes the workspace. Keyboard navigation, visible focus, responsive layouts and reduced-motion support are baseline requirements. Core UI assets must not require internet access.

## Architecture boundaries

- web/: React/TypeScript UI and browser media lifecycle.
- backend/: FastAPI application; later shared encoding/inference, dataset and job services.
- shared/: canonical public vocabulary metadata.
- docs/rebuild/: behaviour, validation and delivery contracts.
- existing src/: historical implementation, retained during migration.

The browser sends data only to explicit backend endpoints after capability checks and user actions. CPU is the initial runtime. Long-running training will use a bounded job queue so it cannot block camera/API requests. Network deployment will require a separate HTTPS/authentication design before exposing data or compute controls.
