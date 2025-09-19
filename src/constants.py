# File paths
DATA_DIR = "./data"
VIDEOS_DIR = DATA_DIR + "/videos"
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
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_HIDDEN_SIZE = 124
DEFAULT_DROPOUT = 0.3
DEFAULT_EPOCHS = 30

# Seq2Seq model hyperparameters
SEQ2SEQ_CONFIG = {
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "hidden_dim": 64,
    "num_layers": 2,
    "use_transformer": True,
    "dropout": 0.1,
    "mse_weight": 1.0,
    "smoothness_weight": 0.3,
    "endpoint_weight": 2.0,
    "teacher_forcing_ratio": 0.5,
    "grad_clip": 1.0,
    "log_every": 100,
    "vis_every": 10,
    "early_stopping": True,
    "patience": 20,
    "epochs": 30,
}


# Video Recording settings
FRAME_RATE = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480