from typing import List, Tuple
import numpy as np

def compute_ausuc(logits: List[List[float]], targets: List[int], seen_classes_mask: List[bool], return_accs: bool=False) -> Tuple[float, Tuple[List[float], List[float]]]:
    """
    Computes area under Seen-Unseen curve (https://arxiv.org/abs/1605.04253)

    :param logits: predicted logits of size [DATASET_SIZE x NUM_CLASSES]
    :param targets: targets of size [DATASET_SIZE]
    :param seen_classes_mask: mask, indicating seen classes of size [NUM_CLASSES]

    :return: AUSUC metric and corresponding curve values
    """

    logits = np.array(logits)
    targets = np.array(targets)
    seen_classes_mask = np.array(seen_classes_mask)
    ds_size, num_classes = logits.shape

    assert len(targets) == ds_size
    assert len(seen_classes_mask) == num_classes

    seen_classes = np.nonzero(seen_classes_mask)[0]
    unseen_classes = np.nonzero(~seen_classes_mask)[0]

    logits_seen = logits[:, seen_classes]
    logits_unseen = logits[:, unseen_classes]

    targets_seen = np.array([next((i for i, t in enumerate(seen_classes) if y == t), -1) for y in targets])
    targets_unseen = np.array([next((i for i, t in enumerate(unseen_classes) if y == t), -1) for y in targets])

    if len(seen_classes) == 0:
        acc = (logits_unseen.argmax(axis=1) == targets_unseen).mean()
        accs_seen = np.array([1., 1., 0.])
        accs_unseen = np.array([0., acc, acc])
    elif len(unseen_classes) == 0:
        acc = (logits_seen.argmax(axis=1) == targets_seen).mean()
        accs_seen = np.array([acc, acc, 0.])
        accs_unseen = np.array([0., 1., 1.])
    else:
        gaps = logits_seen.max(axis=1) - logits_unseen.max(axis=1)
        sorting = np.argsort(gaps)[::-1]
        guessed_seen = logits_seen[sorting].argmax(axis=1) == targets_seen[sorting]
        guessed_unseen = logits_unseen[sorting].argmax(axis=1) == targets_unseen[sorting]

        accs_seen = np.cumsum(guessed_seen) / (targets_seen != -1).sum()
        accs_unseen = np.cumsum(guessed_unseen) / (targets_unseen != -1).sum()
        accs_unseen = accs_unseen[-1] - accs_unseen

        accs_seen = accs_seen[::-1]
        accs_unseen = accs_unseen[::-1]

    auc_score = np.trapz(accs_seen, x=accs_unseen) * 100

    if return_accs:
        return auc_score, (accs_seen, accs_unseen)
    else:
        return auc_score
