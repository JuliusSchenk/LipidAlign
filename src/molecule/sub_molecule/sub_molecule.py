from .constants import *


class SubMolecule:
    """
    This class is a parent class for all submolecules, similar substructures in the molecule.
    Each lipid has at least one of these submolecules and these are just a group of atoms in a specific structure.
    It defines common and abstract functions, which is used to split, optimize and sort Submolecules.
    """

    def __new__(cls, mol, hit_atoms, hit_str):
        # iterate over all known subclasses
        for sub_class in cls.__subclasses__():
            # check if sub_class can use a given separator
            if sub_class.is_separators(hit_str):
                if sub_class.submolecule_decider(mol, hit_atoms) is not None:
                    return super().__new__(sub_class)
        return None

    def __init__(self, mol, hit_atoms, hit_str):
        self._connection_points = []
        self._hit_str = hit_str
        self._mol = mol
        self._bonds = []
        self._atoms = []
        self.set_atoms(self.submolecule_decider(mol, hit_atoms))
        self._generate_bonds_list()
        self._is_assigned = False

    @staticmethod
    def submolecule_decider(mol: any, hit: str) -> list[int]:
        """
        This functions decides, if for a given hit, the submolecule specific structure is found.
        Every child class needs to implement, this logic.
        :param mol: rdkit Molecule object
        :param hit: a string which is in SMARTS format and is a seed point for search
        :return: A list of atom indices
        """
        raise NotImplementedError

    @classmethod
    def get_hit_str(cls) -> list[str]:
        """
        This function returns the child class specific seed points.
        :return: A list of SMARTS strings
        """
        raise NotImplementedError

    def _generate_bonds_list(self):
        """
        Generates a list of bond indices.
        Each child class needs to implement their specific logic.
        :return: None
        """
        self.reset_bonds()
        raise NotImplementedError

    def sort_connections(self, edge_size_dict: dict[int, (int,)]):
        """
        This function sorts the connection points of this Submolecule.
        The sorting is based on position and size of the connected Submolecule(s).
        :param edge_size_dict: A map between atom_idx/connection_point and size & edge
        :return: A sorted list of edges.
        """
        raise NotImplementedError

    @staticmethod
    def _sort_connections_helper(elem):
        """
        Utility function for sort_connections
        """
        return elem[0]

    @classmethod
    def get_all_hit_str(cls) -> list[str]:
        """
        This function collects all separators/seed points of the child classes and removes duplicates.
        :return: A list of all unique separators sorted by size
        """
        seps = []
        for sub_class in cls.__subclasses__():
            seps = seps + sub_class.get_hit_str()

        # remove duplicates
        return sorted(list(dict.fromkeys(seps)), key=len, reverse=True)

    @classmethod
    def is_separators(cls, hit_str: str) -> bool:
        """
        This utility function checks if a given separator is allowed by that child class.
        :param hit_str: SMARTS string
        :return: bool
        """
        return hit_str in cls.get_hit_str()

    def get_color(self, colors=({}, {}), atom_col=(0, 1, 1), bond_col=(0, 1, 1)) -> tuple[dict, dict]:
        for atom_idx in self._atoms:
            if atom_idx not in colors[0]:
                colors[0][atom_idx] = []
            colors[0][atom_idx].append(atom_col)

        for bond_idx in self._bonds:
            if bond_idx not in colors[1]:
                colors[1][bond_idx] = []
            colors[1][bond_idx].append(bond_col)
        return colors

    def get_name(self):
        """
        :return: The name of the chemical substructure
        """
        return self.__class__.__name__

    def get_type(self):
        """
        :return: The name of the class
        """
        return self.__class__.__name__

    def get_bonds(self) -> list:
        """
        :return: A list of bond indices
        """
        return self._bonds

    def reset_bonds(self):
        """
        This function overwrites the bonds list with an empty list
        :return: None
        """
        self._bonds = []

    def get_atoms(self) -> list:
        """
        :return: A list of atom indices
        """
        return self._atoms

    def set_atoms(self, atoms: list[int]):
        """
        This function overwrites this list of atom indices
        :param atoms: new list of atom indices
        :return: None
        """
        self._atoms = atoms

    def get_num_of_carbons(self):
        atoms = [self._mol.GetAtomWithIdx(atom_idx) for atom_idx in self.get_atoms()]
        num_of_carbon = 0
        for atom in atoms:
            if atom.GetSymbol() == "C":
                num_of_carbon += 1
        return num_of_carbon

    def optimize(self):
        """
        This function implements the logic to remove Atoms which are in multiple Submolecules of the same type.
        Each Submolecule might have a different definition of what atom to exclude.
        :return: The edited class if it has any Atoms left
        """

        for atom in [self._mol.GetAtomWithIdx(atom_idx) for atom_idx in self.get_atoms()]:
            if atom.HasProp(self.get_type()):
                return None
        return self.__class__

    def set_assignment(self):
        """
        This function adds for each atom index a property.
        This property represents in how many Submolecules an atom is.
        :return: None
        """
        if not self._is_assigned:
            for atom in [self._mol.GetAtomWithIdx(atom_idx) for atom_idx in self.get_atoms()]:
                atom.SetBoolProp(self.get_type(), True)
                if not atom.HasProp(CONNECTIONPOINT):
                    atom.SetIntProp(CONNECTIONPOINT, 1)
                else:
                    prop = atom.GetIntProp(CONNECTIONPOINT) + 1
                    atom.SetIntProp(CONNECTIONPOINT, prop)
            self._is_assigned = True

    def get_con_points(self):
        """
        This function iterates through every atom index and searches for those who are in more than one Submolecule.
        These Atoms are connection points between two (or more) Submolecules.
        :return: A list of atom indices which are in more than one Submolecule.
        """
        if self._connection_points:
            return self._connection_points
        for atom in [self._mol.GetAtomWithIdx(atom_idx) for atom_idx in self.get_atoms()]:
            if not atom.HasProp(CONNECTIONPOINT):
                raise AttributeError
            if atom.GetIntProp(CONNECTIONPOINT) > 1:
                self._connection_points.append(atom.GetIdx())

        return self._connection_points

    def get_con_point_metric(self):
        """

        :return:
        """
        raise NotImplementedError

    def __hash__(self):
        return super.__hash__(self)

    def __lt__(self, other):
        return len(self.get_atoms()) < len(other.get_atoms())

    def __gt__(self, other):
        return len(self.get_atoms()) > len(other.get_atoms())
