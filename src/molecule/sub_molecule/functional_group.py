from rdkit.Chem.rdchem import Mol

from .sub_molecule import SubMolecule


class FunctionalGroup(SubMolecule):
    """
    This child class represents small unique SubMolecules, which are normally connect aliphatic SubMolecules or
    terminates them.
    The matched substructures doesn't fully match the definition in organic chemistry.
    """
    COLOR = (1, 0, 0)
    _NAMEDICT = {
        "O=[C,c]": "Ketone",
        "OO[C,c]": "Hydroperoxyl",
        "[O;D1][C,c]": "Hydroxyl",
        "C[O;!R]C": "Ether",
        "[C][O;!R]C=O": "Ester",
        "[O;!R]C=O": "Carboxylic Acid",
        "[N;!R]C=O": "Amide",
        "[N;!R][C,c]": "Amine",
        "[C,c][N;!R]~[C,c]": "Amine",
        "[C,c][N;!R]([C,c])[C,c]": "Amine",
        "[C][N;!R]C=O": "Amide",
        "CN(C)(C)C": "Tetramethylammonium",
        "[C][S;!R]C=O": "Thioester",
        "[C,c]~S~[C,c]": "Thioether",
        "OP(=O)(O)O": "Phosphate",
        "OP(=O)(O)OC": "Phosphate",
        "COP(=O)(O)OC": "Phosphate",
        "OS(=O)(=O)O": "Sulfate",
        "OS(=O)(=O)OC": "Sulfate",
        "COS(=O)(=O)OC": "Sulfate",
        "OS(=O)(=O)C": "Sulfite",
    }
    _POLYATOMICSEPS = ["COP(=O)(O)OC", "OP(=O)(O)OC", "OP(=O)(O)O",  # Phosphates
                       "OS(=O)(=O)OC", "COS(=O)(=O)OC", "OS(=O)(O)O", "OS(=O)(=O)C"]  # Sulfates / Sulfite

    def __init__(self, mol: Mol, hit_atoms: tuple[int], hit_str: str):
        super().__init__(mol, hit_atoms, hit_str)

    @staticmethod
    def submolecule_decider(mol: Mol, hits: tuple[int]) -> list[int]:
        # Constrains are defined by separators.
        return list(hits)

    @classmethod
    def get_hit_str(cls) -> list[str]:
        return ["O=[C,c]", "[O;D1][C,c]", "C[O;!R]C", "[O;!R]C=O", "[C][O;!R]C=O", "OO[C,c]",
                "[N;!R][C,c]", "[C,c][N;!R]~[C,c]", "[N;!R]C=O", "[C][N;!R]C=O", "CN(C)(C)C", "[C,c][N;!R]([C,c])[C,c]",
                "[C][S;!R]C=O", "[C,c]~S~[C,c]"] + FunctionalGroup._POLYATOMICSEPS

    def get_name(self) -> str:
        return FunctionalGroup._NAMEDICT[self._hit_str]

    def _generate_bonds_list(self):
        if self._hit_str in FunctionalGroup._POLYATOMICSEPS:
            if len(self._atoms) <= len("COPOOOC"):
                offset = 0
                # C0O1P/S2(=O3)(O4)O5C6
                if len(self._atoms) == len("COPOOOC"):
                    offset = 1
                    self._bonds.append(self._mol.GetBondBetweenAtoms(self._atoms[0], self._atoms[1]).GetIdx())
                # O0P/S1(=O2)(O3)O4C5
                self._bonds.append(
                    self._mol.GetBondBetweenAtoms(self._atoms[0 + offset], self._atoms[1 + offset]).GetIdx())
                self._bonds.append(
                    self._mol.GetBondBetweenAtoms(self._atoms[2 + offset], self._atoms[1 + offset]).GetIdx())
                self._bonds.append(
                    self._mol.GetBondBetweenAtoms(self._atoms[3 + offset], self._atoms[1 + offset]).GetIdx())
                self._bonds.append(
                    self._mol.GetBondBetweenAtoms(self._atoms[4 + offset], self._atoms[1 + offset]).GetIdx())
                if len(self._atoms) >= len("OPOOOC"):
                    self._bonds.append(
                        self._mol.GetBondBetweenAtoms(self._atoms[5 + offset], self._atoms[4 + offset]).GetIdx())
        elif self._hit_str == "CN(C)(C)C":
            # C0N1(C2)(C3)C4
            self._bonds.append(self._mol.GetBondBetweenAtoms(self._atoms[0], self._atoms[1]).GetIdx())
            self._bonds.append(self._mol.GetBondBetweenAtoms(self._atoms[2], self._atoms[1]).GetIdx())
            self._bonds.append(self._mol.GetBondBetweenAtoms(self._atoms[3], self._atoms[1]).GetIdx())
            self._bonds.append(self._mol.GetBondBetweenAtoms(self._atoms[4], self._atoms[1]).GetIdx())
        elif self._hit_str == "[C,c][N;!R]([C,c])[C,c]":
            # C0N1(C2)C3
            self._bonds.append(self._mol.GetBondBetweenAtoms(self._atoms[0], self._atoms[1]).GetIdx())
            self._bonds.append(self._mol.GetBondBetweenAtoms(self._atoms[2], self._atoms[1]).GetIdx())
            self._bonds.append(self._mol.GetBondBetweenAtoms(self._atoms[3], self._atoms[1]).GetIdx())
        else:
            # Linear SubMolecules
            for idx in range(len(self._atoms) - 1):
                bond = self._mol.GetBondBetweenAtoms(self._atoms[idx], self._atoms[idx + 1])
                self._bonds.append(bond.GetIdx())

    def optimize(self):
        """
        No Atom is allowed to be in two or more SubMolecules.
        This implementation ignore carbons, because these are the connection points to other SubMolecules.
        """
        for atom in [self._mol.GetAtomWithIdx(atom_idx) for atom_idx in self.get_atoms()]:
            if atom.GetSymbol() != "C":
                if atom.HasProp(self.get_type()):
                    return None
        return self.__class__

    def sort_connections(self, edge_size_dict: dict[int, list[(int,)]]):
        """
        This function sorts based on the size of the connection_point.
        FunctionalGroup have 1 or 2 connection_points.
        """
        con_points = self.get_con_points().copy()
        if len(con_points) == 1:
            # Nothing to sort except edges on this connection_point
            edge_size_dict[con_points[0]].sort(key=self._sort_connections_helper, reverse=True)
            return [edge for _, edge in edge_size_dict[con_points[0]]]

        elif len(con_points) == 2:
            # sort con_points based on size of connected Submolecules
            end_point_sum = [
                sum([size[0] for size in edge_size_dict[con_points[0]]]),
                sum([size[0] for size in edge_size_dict[con_points[1]]])
            ]
            if end_point_sum[0] < end_point_sum[1]:
                # the last position has (a) bigger Submolecule(s)
                con_points.reverse()

            sorted_edges = []
            for con_point in con_points:
                sorted_edges += [edge for _, edge in edge_size_dict[con_point]]

            return sorted_edges
        else:
            # There should be no cases, where it ends up in here
            raise ValueError

    CON_POINT_METRIC = {
        "O=[C,c]": 0,
        "[O;D1][C,c]": 0,
        "C[O;!R]C": 2,
        "[O;!R]C=O": 1,
        "[C][O;!R]C=O": 2,
        "OO[C,c]": 0,
        "[N;!R][C,c]": 0,  # could be either x-nc-x or nc-(x)-x
        "[C,c][N;!R]~[C,c]": 2,
        "[N;!R]C=O": 1,
        "[C][N;!R]C=O": 2,
        "CN(C)(C)C": 0,
        "[C,c][N;!R]([C,c])[C,c]": 0,
        "[C][S;!R]C=O": 2,
        "[C,c]~S~[C,c]": 2,
        "OP(=O)(O)O": 0,
        "OP(=O)(O)OC": 0,
        "COP(=O)(O)OC": 4,
        "OS(=O)(=O)O": 0,
        "OS(=O)(=O)OC": 0,
        "COS(=O)(=O)OC": 4,
        "OS(=O)(=O)C": 0,
    }

    def get_con_point_metric(self):
        mean_distance = 0
        if len(self.get_con_points()) < 2:
            # there is no distance between less than 2 connection points
            return mean_distance

        mean_distance = FunctionalGroup.CON_POINT_METRIC[self._hit_str]
        return mean_distance
