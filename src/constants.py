from pathlib import Path

# Base Directories
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
VIDEOS_DIR = DATA_DIR / "videos"
OUTPUT_DIR = ROOT_DIR / "output"
LANDMARKS_DIR = DATA_DIR / "landmarks"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DEPENDENCY_DIR = MODELS_DIR / "dependencies"
MODELS_TRAINED_DIR = MODELS_DIR / "trained_models"

# Model Paths
GESTURE_MODEL_PATH = MODELS_TRAINED_DIR / "gesture_model.pth"
MEDIAPIPE_HAND_LANDMARKER_PATH = MODELS_DEPENDENCY_DIR / "hand_landmarker.task"
MEDIAPIPE_POSE_LANDMARKER_PATH = MODELS_DEPENDENCY_DIR / "pose_landmarker_lite.task"

# Metadata Paths
LANDMARKS_DIR_METADATA_PKL = LANDMARKS_DIR / "landmarks_metadata.pkl"
GESTURE_MODEL_METADATA_PATH = MODELS_TRAINED_DIR / "gesture_model_metadata.pkl"

# Gesture recognizer model hyperparameters
DEFAULT_SEQUENCE_LENGTH = 30
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_HIDDEN_SIZE = 124
DEFAULT_DROPOUT = 0.1
DEFAULT_EPOCHS = 30

# Seq2Seq model hyperparameters
SEQ2SEQ_CONFIG = {
    # Training hyperparameters
    "epochs": 50,
    "batch_size": 32,
    # Architecture hyperparameters
    "hidden_dim": 64,
    "num_layers": 1,
    "noise_dim": 100,
    "dropout": 0.3,
    # GAN training hyperparameters
    "generator_lr": 0.1,
    "discriminator_lr": 0.0001,
    "weight_decay": 1e-3,
    "beta1": 0.5,
    "beta2": 0.999,
    # Training strategy
    "label_smoothing": 0.1,
    "generator_steps": 1,
    "discriminator_steps": 3, # train once every "this number" epochs
    # Dataset parameters
    "gap_size": 3,
    "max_samples": 10000,
    "cache_size": 50,
    # Logging
    "log_every": 1,
}

# Video Recording settings
FRAME_RATE = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
