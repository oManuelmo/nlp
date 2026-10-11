import random
import numpy as np
import torch

SEED = 42

MODEL_1 = "Musixmatch/umberto-commoncrawl-cased-v1"
MODEL_2 = "FacebookAI/xlm-roberta-base"

MODEL_NAMES = {
    "umberto": MODEL_1,
    "xlmr": MODEL_2,
}

MAX_LENGTH = 256
LEARNING_RATE = 2e-5
BATCH_SIZE = 8
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01

LABEL_NAMES = [
    "O",
    "B-LOC", "I-LOC",
    "B-ORG", "I-ORG",
    "B-PER", "I-PER",
]

LABEL2ID = {
    label: i for i, label in enumerate(LABEL_NAMES)
}

ID2LABEL = {
    i: label for label, i in LABEL2ID.items()
}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
