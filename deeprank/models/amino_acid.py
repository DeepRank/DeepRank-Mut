

class AminoAcid:
    def __init__(self, name, code, letter):
        self.name = name
        self.code = code
        self.letter = letter

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)
