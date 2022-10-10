from deeprank.models.polarity import Polarity


class AminoAcid:
    def __init__(self, name: str, code: str, letter: str,
                 charge: float, polarity: Polarity, size: int,
                 count_hydrogen_bond_donors: int,
                 count_hydrogen_bond_acceptors: int):

        self.name = name
        self.code = code
        self.letter = letter

        # these settings apply to the side chain
        self._size = size
        self._charge = charge
        self._polarity = polarity
        self._count_hydrogen_bond_donors = count_hydrogen_bond_donors
        self._count_hydrogen_bond_acceptors = count_hydrogen_bond_acceptors

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    @property
    def count_hydrogen_bond_donors(self) -> int:
        return self._count_hydrogen_bond_donors

    @property
    def count_hydrogen_bond_acceptors(self) -> int:
        return self._count_hydrogen_bond_acceptors

    @property
    def charge(self) -> float:
        return self._charge

    @property
    def polarity(self) -> Polarity:
        return self._polarity

    @property
    def size(self) -> int:
        return self._size

