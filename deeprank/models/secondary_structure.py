from enum import Enum

import numpy


class SecondaryStructure(Enum):
    HELIX = 1
    STRAND = 2
    OTHER = 0

    def one_hot(self):
        if self == SecondaryStructure.HELIX:
            return numpy.array([1, 0])

        elif self == SecondaryStructure.STRAND:
            return numpy.array([0, 1])

        else:
            return numpy.array([0, 0])
