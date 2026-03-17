# Sign Language Translation System

A real-time bidirectional translation system between speech and **Namibian Sign Language (NSL)**, combining speech recognition, computer vision, and avatar-based visualization.

<!-- Going, to add a demo here, currently in progress -->

## Key Features

* 🎤 **Speech → Sign Translation**: Convert spoken English into sign language visualization
* 📷 **Sign → Text Recognition**: Real-time sign detection using computer vision
* 🔁 **Bidirectional Pipeline**: Full loop from speech ↔ text ↔ sign
* 🖥️ **Unified Interface**: Single desktop app (PyQt6) integrating all components
* ⚙️ **Modular Design**: Six independent but connected processing stages

---

## Architecture

The system is composed of six modular components:

| Component             | Description                          | Status |
| --------------------- | ------------------------------------ | ------ |
| Speech → Text         | Real-time speech transcription       | ✅      |
| Text → Gloss          | English to NSL gloss translation     | ✅      |
| Gloss → Visualization | Avatar-based sign rendering          | ✅      |
| Video → Gloss         | Sign recognition via computer vision | ⚠️     |
| Gloss → Text          | Gloss to English translation         | ⚠️     |
| Text → Speech         | Speech synthesis                     | ✅      |

### Translation Flow

* **Forward**: Speech → Text → Gloss → Visualization
* **Reverse**: Video → Gloss → Text → Speech

---

## Installation

```bash
git clone https://github.com/RighteousW/sign_avatar.git
cd sign_avatar
pip install -r requirements.txt
```

---

## Usage

```bash
pip install -e .
unified-demo
```

This launches a desktop application with:

* Speech-to-sign translation
* Sign-to-text recognition
* Real-time visualization interface

> ⚠️ Note: Speech-to-text and text-to-speech require an internet connection.

---

## Technical Highlights

* Designed and implemented a **multi-stage ML pipeline** integrating speech, vision, and language components
* Built a **modular architecture** to allow independent development and testing of each stage
* Addressed **real-world deployment challenges**, including:

  * Temporal segmentation in continuous signing
  * Distribution mismatch between pipeline components
  * Error propagation across stages

---

## Challenges & Limitations

### Video → Gloss (Sign Recognition)

* Achieves **96% accuracy** on isolated test data
* Struggles with **continuous signing** due to unclear sign boundaries
* Sensitive to lighting, camera angle, and real-world variability

### Gloss → Text

* Trained on **synthetic data with constrained vocabulary**
* Limited generalization to noisy or real-world gloss sequences

### System-Level Challenges

* Mismatch between independently trained components
* Errors compound across the pipeline (e.g., noisy vision output affects translation quality)

---

## Future Work

* Improve **temporal segmentation** (e.g., sliding window or sequence models)
* Expand gloss-to-text with real-world datasets
* Apply **data augmentation** for robustness (lighting, angles, motion)
* Explore **attention-based models** for better sequence understanding
* Investigate **end-to-end training** to reduce pipeline mismatch

---

## Tech Stack

* **Languages**: Python
* **Libraries/Frameworks**: OpenCV, PyQt6, (TensorFlow / PyTorch if applicable)
* **Tools**: Git, virtual environments

---

## Project Structure

```bash
sign_avatar/
├── audio2text/
├── text2gloss/
├── gloss2viz/
├── video2gloss/
├── gloss2text/
├── text2speech/
├── app/
└── ...
```

---

## Requirements

* Python 3.12+
* PyQt6
* OpenCV
* See `requirements.txt` for full dependencies

---

## Acknowledgments

Developed as a final-year project focused on real-world sign language translation challenges.
Some architectural ideas were informed by existing research and publicly available ASL datasets.

---

## Author

**Righteous Wasambo**
