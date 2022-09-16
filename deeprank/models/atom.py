

class Atom:
    def __init__(self, id_, position, chain_id, name, element, residue=None, altloc=None, occupancy=None, b_factor=None):
        self.id = id_
        self.name = name
        self.element = element
        self.chain_id = chain_id
        self.position = position
        self.residue = residue
        self.altloc = altloc
        self.occupancy = occupancy
        self.b_factor = b_factor

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __lt__(self, other):
        return self.id < other.id

    def __repr__(self):
        return "Atom {} ({}) from {} {} at {}".format(self.id, self.name, self.chain_id, self.residue, self.position)

