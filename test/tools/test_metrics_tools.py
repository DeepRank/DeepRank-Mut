import torch

from deeprank.models.variant import VariantClass
from deeprank.tools.metrics import get_labels_from_output, get_labels_from_targets


def test_labels_from_output():
    data = torch.tensor([[0.1, -0.1],
                         [0.0, 1.1],
                         [1.2, -0.1]])

    labels = get_labels_from_output(data)

    assert labels == [VariantClass.UNKNOWN,
                      VariantClass.PATHOGENIC,
                      VariantClass.BENIGN], f"labels are {labels}"


def get_labels_from_targets():
    data = torch.tensor([[0], [1], [1], [0], [0]])

    labels = get_labels_from_targets(data)

    assert labels == [VariantClass.BENIGN,
                      VariantClass.PATHOGENIC,
                      VariantClass.PATHOGENIC,
                      VariantClass.BENIGN,
                      VariantClass.BENIGN], f"labels are {labels}"
