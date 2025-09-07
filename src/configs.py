from .utils import create_directories
import os


# Training configurations
OUTPUT_UNITS = 38
NUM_UNITS = [256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 90
BATCH_SIZE = 64
MODEL_OUTPUT_PATH = os.path.join("models")
create_directories(MODEL_OUTPUT_PATH)
SAVE_MODEL_PATH = os.path.join(MODEL_OUTPUT_PATH, "model.h5")


# Preprocessing configurations
DATA_SET_PATH = os.path.join("dataset")

KERN_DATASET_PATH = os.path.join(DATA_SET_PATH, "deutschl", "erk")
SAVE_DIR = os.path.join(DATA_SET_PATH, "melodies")
create_directories(SAVE_DIR)

SINGLE_FILE_DATASET = os.path.join(DATA_SET_PATH, "file_dataset")
create_directories(SINGLE_FILE_DATASET)


MAPPING_PATH = os.path.join(DATA_SET_PATH, "mapping.json")
SEQUENCE_LENGTH = 64

# durations are expressed in quarter length
ACCEPTABLE_DURATIONS = [
    0.25,  # 16th note
    0.5,  # 8th note
    0.75,
    1.0,  # quarter note
    1.5,
    2,  # half note
    3,
    4,  # whole note
]


# Melody configurations

MELODY_OUTPUT_PATH = os.path.join("outputs", "mel.mid")
create_directories(MELODY_OUTPUT_PATH)
