import math

from .sub_molecule import SubMolecule


class AliphaticChain(SubMolecule):
    """
    This child class represents aliphatic chain, which have no branching, but connections to other chains.
    These chains only contain carbon and are acyclic.
    """
    COLOR = (0, 0, 1)
    MIN_CHAIN_LENGTH = 2

    def __init__(self, mol, hit_atoms, hit_str):
        super().__init__(mol, hit_atoms, hit_str)

    @staticmethod
    def submolecule_decider(mol, hit) -> list[int]:
        """
        This function returns a list of atom indices representing an alkyl-chain for a given hit.
        :param mol:rdkit molecule object
        :param hit: tuple of two atom indices, which defines the direction of search (idx_0 -> idx_1)
        :return: List of indices ending with a connected func. group
        """
        first_atom = mol.GetAtomWithIdx(hit[0])
        # First atom is a methyl-group
        if first_atom.GetSymbol() == "C" and first_atom.GetDegree() == 1:
            atoms = AliphaticChain.terminating_chain_decider_rec(mol, hit)
            atoms.reverse()
        else:
            atoms = AliphaticChain.interchain_decider_rec(mol, hit)

        # remove every non-carbon at the end
        if len(atoms) > 2:
            for idx in [-1, 0]:
                atom = mol.GetAtomWithIdx(atoms[idx])
                if atom.GetSymbol() != "C" and not atom.IsInRing():
                    atoms.pop(idx)

        return atoms if len(atoms) >= AliphaticChain.MIN_CHAIN_LENGTH else None

    @staticmethod
    def terminating_chain_decider_rec(mol, hit) -> list[int]:
        """
        Recursive depth first search for longest chain, which starts at a methyl-group.
        :param mol: rdkit molecule object
        :param hit: tuple of two atom indices, which defines the direction of search (idx_0 -> idx_1)
        :return:  idx of the longest chain, which stops at the first cyclic or non-aliphatic atom
        """
        atoms = [hit[0]]
        current_atom = mol.GetAtomWithIdx(hit[1])
        prev_atom_idx = hit[0]

        while current_atom.GetSymbol() == "C":
            if current_atom.IsInRing():
                atoms.append(current_atom.GetIdx())
                return atoms

            # getting neighbors and removing previous atom
            neighbors_idx = [n.GetIdx() for n in current_atom.GetNeighbors()]
            neighbors_idx.remove(prev_atom_idx)

            # reached a dead end
            if len(neighbors_idx) == 0:
                return []

            # only one direction, continue down chain
            elif len(neighbors_idx) == 1:

                # aliphatic carbon, therefore we continue down the chain
                atoms.append(current_atom.GetIdx())
                prev_atom_idx = current_atom.GetIdx()
                neighbor_atom = mol.GetAtomWithIdx(neighbors_idx[0])
                current_atom = neighbor_atom

            # intersection
            else:
                # depth first search
                rec_atoms = []
                for neighbor_idx in neighbors_idx:
                    next_hit = (current_atom.GetIdx(), neighbor_idx)
                    rec_atoms.append(AliphaticChain.terminating_chain_decider_rec(mol, next_hit))

                # get longest chain
                if rec_atoms:
                    new_atoms = max(rec_atoms, key=len)
                    atoms += new_atoms
                    if new_atoms:
                        return atoms
                    else:
                        return []
                else:
                    return []

        return atoms

    @staticmethod
    def interchain_decider_rec(mol, hit) -> list[int]:
        """
        Recursive depth first search for highest degree (at the ends) chain.
        :param mol: rdkit molecule object
        :param hit: tuple of two atom indices, which defines the direction of search (idx_0 -> idx_1)
        :return:  idx of the highest degree chain, preferring longer chains
        """

        atoms = [hit[0]]
        current_atom = mol.GetAtomWithIdx(hit[1])
        prev_atom_idx = hit[0]
        # after an intersection, where carbon is not the current atom
        if current_atom.GetSymbol() != "C":
            atoms.append(hit[1])
            return atoms

        while current_atom.GetSymbol() == "C":
            if current_atom.IsInRing():
                # check if previous and current atom are in separate rings
                prev_atom = mol.GetAtomWithIdx(prev_atom_idx)
                if prev_atom.IsInRing() and not mol.GetBondBetweenAtoms(current_atom.GetIdx(),
                                                                        prev_atom_idx).IsInRing():
                    atoms.append(current_atom.GetIdx())
                return atoms

            # getting neighbors and removing previous atom
            neighbors_idx = [n.GetIdx() for n in current_atom.GetNeighbors()]
            neighbors_idx.remove(prev_atom_idx)

            # reached a dead end
            if len(neighbors_idx) == 0:
                return []

            # only one direction, continue down chain
            elif len(neighbors_idx) == 1:

                # check if neighbor atom is not an aliphatic carbon
                neighbor_atom = mol.GetAtomWithIdx(neighbors_idx[0])
                atoms.append(current_atom.GetIdx())
                if neighbor_atom.GetSymbol() != "C" or neighbor_atom.IsInRing():
                    atoms.append(neighbor_atom.GetIdx())
                    return atoms

                # aliphatic carbon, therefore we continue down the chain
                prev_atom_idx = current_atom.GetIdx()
                current_atom = neighbor_atom

            # intersection
            else:
                # depth first search
                rec_atoms = []
                for neighbor_idx in neighbors_idx:
                    next_hit = (current_atom.GetIdx(), neighbor_idx)
                    rec_atoms.append(AliphaticChain.interchain_decider_rec(mol, next_hit))

                # get longest chain
                if rec_atoms:
                    new_atoms = max(rec_atoms, key=len)
                    atoms += new_atoms
                    if new_atoms:
                        return atoms
                    else:
                        return []
                else:
                    return []

        # recursion exit
        return atoms

    def _generate_bonds_list(self):
        self.reset_bonds()
        for idx in range(len(self._atoms) - 1):
            bond = self._mol.GetBondBetweenAtoms(self._atoms[idx], self._atoms[idx + 1])
            self._bonds.append(bond.GetIdx())

    @classmethod
    def get_hit_str(cls) -> list[str]:
        """
        This function returns the Smarts string, which represents the separators for this submolecule.
        """
        return ["[C;D1]~*",  # terminus Carbon / methyl
                "[C;R]~[C;!R]",  # chain connected to a ring
                "[!C]~[C,c]",  # chain connected to a nonmetal
                ]

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

    def optimize(self):
        """
        This function shortens chains, when atoms are in multiple chains.
        """
        # func. group moving to C-terminus
        atoms_idx = [atom_idx for atom_idx in self.get_atoms()]
        last_unassigned = 0
        for index, atom_idx in enumerate(atoms_idx):
            if self._mol.GetAtomWithIdx(atom_idx).HasProp(self.get_type()):
                last_unassigned = index

        # there is overlap
        if last_unassigned:
            atoms_idx = atoms_idx[last_unassigned:]
            # the overlap is too big
            if len(atoms_idx) < AliphaticChain.MIN_CHAIN_LENGTH:
                return None
            self.set_atoms(atoms_idx)
            self._generate_bonds_list()

        return self.__class__

    def sort_connections(self, edge_size_dict: dict[int, list[(int,)]]):
        """
        This function sorts based on the size of the connection_point on either end of the chain.
        """
        # Check how many connection_points are at the end of this chain
        con_points = self.get_con_points().copy()
        end_points = [self.get_atoms()[0], self.get_atoms()[-1]]

        for end_idx in range(len(end_points)):
            if end_points[end_idx] not in con_points:
                end_points.pop(end_idx)

        if len(end_points) > 2:
            # There should be no cases, where it ends up in here
            raise ValueError

        if len(end_points) == 2:
            end_point_sum = [
                sum([size[0] for size in edge_size_dict[end_points[0]]]),
                sum([size[0] for size in edge_size_dict[end_points[1]]])
            ]
            if end_point_sum[0] < end_point_sum[1]:
                # the last position has (a) bigger Submolecule(s)
                con_points.reverse()

        elif len(end_points) == 1:
            # check if this atom_idx is in first or last position
            end_point = end_points[0]
            if con_points.index(end_point) != 0:
                # if not first then we reverse con_points, so an end_point is in first position
                con_points.reverse()

        # generate sorted_edges based on the order of con_points
        sorted_edges = []
        for con_point in con_points:
            # sorts edges on same con_point
            edge_size_dict[con_point].sort(key=self._sort_connections_helper, reverse=True)
            for _, edge in edge_size_dict[con_point]:
                sorted_edges.append(edge)
        return sorted_edges

    def __lt__(self, other):
        # if equal len, then non-terminating is longer
        if len(self.get_atoms()) == len(other.get_atoms()):
            return self._mol.GetAtomWithIdx(self.get_atoms()[-1]).GetDegree() < other._mol.GetAtomWithIdx(
                other.get_atoms()[-1]).GetDegree()
        return len(self.get_atoms()) < len(other.get_atoms())

    def __gt__(self, other):
        if len(self.get_atoms()) == len(other.get_atoms()):
            return self._mol.GetAtomWithIdx(self.get_atoms()[-1]).GetDegree() > other._mol.GetAtomWithIdx(
                other.get_atoms()[-1]).GetDegree()
        return len(self.get_atoms()) > len(other.get_atoms())

    def get_con_point_metric(self):
        mean_distance = 0
        if len(self.get_con_points()) < 2:
            # there is no distance between less than 2 connection points
            return mean_distance

        con_p = self.get_con_points().copy()
        if self.get_atoms()[-1] not in con_p:
            con_p.append(self.get_atoms()[-1])

        # the metric is the square root of the distances squared
        prev_idx = None
        for cp in con_p:
            if prev_idx is not None:
                current_idx = self.get_atoms().index(cp)
                mean_distance += pow(current_idx - prev_idx, 2)
                prev_idx = current_idx
            else:
                prev_idx = self.get_atoms().index(cp)

        return math.sqrt(mean_distance / len(self.get_con_points()))
