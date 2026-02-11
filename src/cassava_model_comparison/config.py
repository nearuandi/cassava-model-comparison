from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train_images"
RUNS_DIR = PROJECT_ROOT / "runs"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
# 0.0001
LEARNING_RATE = 1e-4

NUM_CLASSES = 5
NUM_WORKERS = 4
SEED = 42

LR_PATIENCE = 3
LR_FACTOR = 0.5
LR_MODE = "min"

CLASS_NAMES = [
               "CBB (Cassava Bacterial Blight)",
               "CBSD (Cassava Brown Streak Disease)",
               "CGM (Cassava Green Mottle)",
               "CMD (Cassava Mosaic Disease)",
               "Healthy"
]

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD  = [0.229, 0.224, 0.225]