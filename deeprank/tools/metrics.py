from typing import List, Optional

import torch

from deeprank.models.variant import VariantClass


def get_labels_from_probabilities(probabilities: torch.tensor,
                                  unknown_treshold: Optional[float] = 0.5) -> List[VariantClass]:
    """
    Args:
        probabilities: [x, 2] considered negative if left value > right value
        unknown_treshold: if the values are both below this value, then consider the class UNKNOWN
    """

    total = probabilities.shape[0]

    labels = []
    for index in range(total):
        if probabilities[index, 0] < unknown_treshold and probabilities[index, 1] < unknown_treshold:
            label = VariantClass.UNKNOWN

        elif probabilities[index, 0] < probabilities[index, 1]:
            label = VariantClass.PATHOGENIC

        else:
            label = VariantClass.BENIGN

        labels.append(label)

    return labels
