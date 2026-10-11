from transformers import AutoModelForTokenClassification

from src.config import ID2LABEL, LABEL2ID, LABEL_NAMES


def get_model(model_name, num_labels=None, id2label=ID2LABEL, label2id=LABEL2ID):
    """
    Initialize and return an AutoModelForTokenClassification instance.
    """
    if num_labels is None:
        num_labels = len(label2id) if label2id else len(LABEL_NAMES)

    return AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
