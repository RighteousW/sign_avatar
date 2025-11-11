# Sign Language Translation System

A bidirectional sign language translation system with six integrated components enabling translation between spoken language and Namibiam Sign Language (NSL).

## Overview

This honors project implements a complete pipeline for sign language translation, supporting multiple translation directions:
- **Forward direction**: Speech → Text → Gloss → Visualization
- **Reverse direction**: Video → Gloss → Text → Speech

## System Architecture

### Components

1. **Speech → Text (Audio2Text)**
   - Real-time audio transcription
   - Converts spoken English to written text

2. **Text → Gloss**
   - Translates English text to NSL gloss notation
   - Gloss represents the grammatical structure of sign language

3. **Gloss → Visualization**
   - Renders sign language glosses as visual representations
   - Displays signs for user comprehension

4. **Video → Gloss**
   - Computer vision-based sign recognition from webcam
   - Real-time detection of signs from video input
   - **Status**: Achieves 96% accuracy on test data; struggles with temporal segmentation in continuous signing

5. **Gloss → Text**
   - Converts NSL gloss notation back to English text
   - **Status**: Trained on synthetic data with constrained vocabulary

6. **Text → Speech**
   - Text-to-speech synthesis
   - Completes the bidirectional translation loop

## Installation

```bash
# Clone the repository
git clone https://github.com/RighteousW/sign_avatar.git
cd sign_avatar

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the unified demo application:

```bash
#install command scripts using setup.py
pip install -e .

unified-demo
```

The application provides a tabbed interface with all six components accessible from a single window.

## Technical Challenges

### Video → Gloss
The primary challenge is **temporal segmentation** in continuous signing:
- Model performs well on isolated, pre-segmented signs (96% test accuracy)
- Struggles with continuous signing where sign boundaries are unclear
- Additional challenges: lighting variations, camera angles, distribution shift

### Gloss → Text
- Trained on synthetic data with clean, well-structured glosses
- Constrained vocabulary to ensure grammatical correctness
- Limited generalization to noisy real-world gloss sequences
- Architecture validated separately using published ASL datasets

### Integration Challenges
- Data distribution mismatch between independently-trained components
- Error propagation through the pipeline (noisy video→gloss output affects gloss→text)
- Highlights the gap between component-level and system-level performance

## Performance

| Component | Status | Notes |
|-----------|--------|-------|
| Speech → Text | ✅ Working | Real-time transcription |
| Text → Gloss | ✅ Working | Reliable translation |
| Gloss → Visualization | ✅ Working | Visual rendering functional |
| Video → Gloss | ⚠️ Limited | 96% test accuracy; deployment challenges |
| Gloss → Text | ⚠️ Limited | Constrained vocabulary; synthetic training |
| Text → Speech | ✅ Working | Audio synthesis functional |

## Future Work

- Improve temporal segmentation for continuous sign recognition e.g. sliding window for inference
- Expand gloss→text vocabulary and real-world training data
- Implement better data augmentation for lighting/angle variations
- Explore attention mechanisms for sign boundary detection
- Develop unified training approach to reduce distribution mismatch

## Requirements

- Python 3.12.3+
- PyQt6
- OpenCV
- [Additional dependencies in requirements.txt]


## Acknowledgments

This project was completed as an honors thesis project. Architecture evaluation used published ASL datasets and benchmarks from the sign language processing research community.

## Author

Righteous Wasambo