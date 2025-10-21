from pathlib import Path

# Base Directories
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
VIDEOS_DIR = DATA_DIR / "gloss_videos"
OUTPUT_DIR = DATA_DIR / "output"
DATASET_DIR = DATA_DIR / "dataset"
LANDMARKS_DIR = DATASET_DIR / "landmarks"
MODELS_DIR = ROOT_DIR / "models"
DEPENDENCY_MODELS_DIR = MODELS_DIR / "dependencies"
TRAINED_MODELS_DIR = MODELS_DIR / "trained_models"
GESTURE_MODEL_DIR = TRAINED_MODELS_DIR / "gesture_recognizer"

# Model Paths
GESTURE_MODEL_0_SKIP = GESTURE_MODEL_DIR / "gesture_model_0_skip.pth"
GESTURE_MODEL_1_SKIP = GESTURE_MODEL_DIR / "gesture_model_1_skip.pth"
GESTURE_MODEL_2_SKIP = GESTURE_MODEL_DIR / "gesture_model_2_skip.pth"

GESTURE_MODEL_2_SKIP_METADATA_PATH = (
    GESTURE_MODEL_DIR / "gesture_model_metadata_2_skip.pkl"
)


GESTURE_MODEL_PATH = GESTURE_MODEL_DIR / "gesture_model_2_skip.pth"
MEDIAPIPE_HAND_LANDMARKER_PATH = DEPENDENCY_MODELS_DIR / "hand_landmarker.task"
MEDIAPIPE_POSE_LANDMARKER_PATH = DEPENDENCY_MODELS_DIR / "pose_landmarker_lite.task"

# Metadata Paths
LANDMARKS_DIR_METADATA_PKL = LANDMARKS_DIR / "landmarks_metadata.pkl"
GESTURE_MODEL_METADATA_PATH = GESTURE_MODEL_DIR / "gesture_model_metadata_2_skip.pkl"
REPRESENTATIVES_LEFT = OUTPUT_DIR / "gesture_metadata" / "representatives_left.json"

# Gesture recognizer model hyperparameters
DEFAULT_SEQUENCE_LENGTH = 30
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_HIDDEN_SIZE = 256
DEFAULT_DROPOUT = 0.1
DEFAULT_EPOCHS = 30

# Video Recording settings
FRAME_RATE = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
