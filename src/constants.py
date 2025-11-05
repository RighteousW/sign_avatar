from pathlib import Path

# Base Directories
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
VIDEOS_DIR = DATA_DIR / "gloss_videos"
OUTPUT_DIR = DATA_DIR / "output"
MODELS_DIR = ROOT_DIR / "models"

# Dataset Directories
DATASET_DIR = DATA_DIR / "dataset"
LANDMARKS_DIR_HANDS_ONLY = DATASET_DIR / "landmarks_hands_only"
LANDMARKS_DIR_HANDS_POSE = DATASET_DIR / "landmarks_hands_pose"

# Model Directories
DEPENDENCY_MODELS_DIR = MODELS_DIR / "dependencies"
TRAINED_MODELS_DIR = MODELS_DIR / "trained_models"
GESTURE_RECOGNIZER_DIR = TRAINED_MODELS_DIR / "gesture_recognizer"
GESTURE_MODEL_HANDS_ONLY_DIR = GESTURE_RECOGNIZER_DIR / "gesture_recognizer_hands"
GESTURE_MODEL_HANDS_POSE_DIR = GESTURE_RECOGNIZER_DIR / "gesture_recognizer_hands_pose"
GLOSS2TEXT_LOGS = TRAINED_MODELS_DIR / "gloss2text_logs"

# Dependency Models
MEDIAPIPE_HAND_LANDMARKER_PATH = DEPENDENCY_MODELS_DIR / "hand_landmarker.task"
MEDIAPIPE_POSE_LANDMARKER_PATH = DEPENDENCY_MODELS_DIR / "pose_landmarker_lite.task"

# Special Paths
GLOSS2TEXT_MODEL_SYNTHETIC = (
    GLOSS2TEXT_LOGS
    / "synthetic_MediTOD_batch-size8_hidden-size256_epochs5_timestamp20251029_133529"
)
GLOSS2TEXT_MODEL_SYNTHETIC_QUANTIZED = GLOSS2TEXT_MODEL_SYNTHETIC / "quantized"
REPRESENTATIVES_LEFT = OUTPUT_DIR / "gesture_metadata" / "representatives_left.json"
LANDMARKS_DIR_METADATA_PKL = LANDMARKS_DIR_HANDS_ONLY / "landmarks_metadata.pkl"


def get_gesture_model_path(use_pose: bool, skip_pattern: int) -> Path:
    """Get gesture model path dynamically
    Args:
        use_pose: True for hands+pose model, False for hands-only
        skip_pattern: 0, 1, or 2
    """
    base_dir = (
        GESTURE_MODEL_HANDS_POSE_DIR if use_pose else GESTURE_MODEL_HANDS_ONLY_DIR
    )
    return base_dir / f"gesture_model_{skip_pattern}_skip.pth"


def get_gesture_metadata_path(use_pose: bool, skip_pattern: int) -> Path:
    """Get gesture model metadata path dynamically
    Args:
        use_pose: True for hands+pose model, False for hands-only
        skip_pattern: 0, 1, or 2
    """
    base_dir = (
        GESTURE_MODEL_HANDS_POSE_DIR if use_pose else GESTURE_MODEL_HANDS_ONLY_DIR
    )
    return base_dir / f"gesture_model_metadata_{skip_pattern}_skip.pkl"

# Hyperparameters
DEFAULT_SEQUENCE_LENGTH = 30
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_HIDDEN_SIZE = 256
DEFAULT_DROPOUT = 0.4
DEFAULT_EPOCHS = 30

# Video Settings
FRAME_RATE = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Create directories if they don't exist
def ensure_dirs():
    """Create all necessary directories"""
    dirs = [
        DATA_DIR,
        VIDEOS_DIR,
        OUTPUT_DIR,
        DATASET_DIR,
        LANDMARKS_DIR_HANDS_ONLY,
        LANDMARKS_DIR_HANDS_POSE,
        MODELS_DIR,
        DEPENDENCY_MODELS_DIR,
        TRAINED_MODELS_DIR,
        GESTURE_RECOGNIZER_DIR,
        GESTURE_MODEL_HANDS_ONLY_DIR,
        GESTURE_MODEL_HANDS_POSE_DIR,
        GLOSS2TEXT_LOGS,
        OUTPUT_DIR / "gesture_metadata",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

# Auto-create directories on import
ensure_dirs()
