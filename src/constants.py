# File paths
DATA_DIR = "./data"
VIDEOS_DIR = DATA_DIR + "/videos"
OUTPUT_DIR = "./output"
LANDMARKS_DIR = DATA_DIR + "/landmarks"
PROCESSED_GESTURE_DATA_PATH = DATA_DIR + "/processed_gesture_data.pkl"

# model paths
MODELS_DIR = "./models"
MODELS_DEPENDENCY_DIR = MODELS_DIR + "/dependencies"
MODELS_TRAINED_DIR = MODELS_DIR + "/trained_models"
MEDIAPIPE_HAND_LANDMARKER_PATH = MODELS_DEPENDENCY_DIR + "/hand_landmarker.task"
MEDIAPIPE_POSE_LANDMARKER_PATH = MODELS_DEPENDENCY_DIR + "/pose_landmarker_lite.task"

# Gesture recognizer model hyperparameters
DEFAULT_SEQUENCE_LENGTH = 30
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_HIDDEN_SIZE = 124
DEFAULT_DROPOUT = 0.5
DEFAULT_EPOCHS = 30

# Seq2Seq model hyperparameters
SEQ2SEQ_CONFIG = {
    # Training hyperparameters
    "epochs": 50,
    "batch_size": 32,
    # Architecture hyperparameters
    "hidden_dim": 128,
    "num_layers": 2,
    "noise_dim": 100,
    "dropout": 0.3,
    # GAN training hyperparameters
    "generator_lr": 0.0003,
    "discriminator_lr": 0.00008,
    "weight_decay": 1e-5,
    "beta1": 0.5,
    "beta2": 0.999,
    # Training strategy
    "label_smoothing": 0.3,
    "generator_steps": 3,
    "discriminator_steps": 1,
    # Dataset parameters
    "gap_size": 5,
    "max_samples": 5000,
    "cache_size": 50,
    # Logging
    "log_every": 5,
}

# Video Recording settings
FRAME_RATE = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
