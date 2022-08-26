from enum import Enum

import numpy


class SecondaryStructure(Enum):
    HELIX = 1
    STRAND = 2
    OTHER = 0

    def one_hot(self):
        if self == SecondaryStructure.HELIX:
            return [1, 0]

        elif self == SecondaryStructure.STRAND:
            return [0, 1]

        return [0, 0]
