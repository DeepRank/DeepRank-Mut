

class Residue:
    def __init__(self, number, name, chain_id, insertion_code=None):
        self.number = number
        self.insertion_code = insertion_code
        self.name = name
        self.chain_id = chain_id
        self.atoms = []

    def __hash__(self):
        return hash((self.chain_id, self.number))

    def __eq__(self, other):
        return self.chain_id == other.chain_id and self.number == other.number

    def __repr__(self):
        return "Residue {} {} of chain {}".format(self.name, self.number, self.chain_id)
