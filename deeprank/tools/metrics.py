from typing import List, Optional

import torch

from deeprank.models.variant import VariantClass


def get_labels_from_output(output_data: torch.Tensor,
                           unknown_treshold: Optional[float] = 0.5) -> List[VariantClass]:
    """
    Args:
        output_data: [x, 2] considered BENIGN if left value > right value and otherwise PATHOGENIC
        unknown_treshold: if the absolute difference between the values is below this value, then consider the class UNKNOWN
    """

    total = output_data.shape[0]

    labels = []
    for index in range(total):
        if abs(output_data[index, 0] - output_data[index, 1]) < unknown_treshold:
            label = VariantClass.UNKNOWN

        elif output_data[index, 0] < output_data[index, 1]:
            label = VariantClass.PATHOGENIC

        else:
            label = VariantClass.BENIGN

        labels.append(label)

    return labels


def get_labels_from_targets(target_data: torch.Tensor) -> List[VariantClass]:
    """
    Args:
        target_data: [x, 1] where 0 means BENIGN and 1 means PATHOGENIC
    """

    total = target_data.shape[0]

    labels = []
    for index in range(total):
        if target_data[index] > 0:
            label = VariantClass.PATHOGENIC
        else:
            label = VariantClass.BENIGN

        labels.append(label)

    return labels
