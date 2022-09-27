import torch

from deeprank.models.variant import VariantClass
from deeprank.tools.metrics import get_labels_from_probabilities


def test_labels_from_probabililties():
    data = torch.tensor([[0.1, -0.1],
                         [0.0, 1.1],
                         [1.2, -0.1]])

    labels = get_labels_from_probabilities(data)

    assert labels == [VariantClass.UNKNOWN,
                      VariantClass.PATHOGENIC,
                      VariantClass.BENIGN], f"labels are {labels}"
