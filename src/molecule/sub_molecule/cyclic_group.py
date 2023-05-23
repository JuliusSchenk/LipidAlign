import math

from rdkit.Chem.rdchem import Mol

from .sub_molecule import SubMolecule


class CyclicGroup(SubMolecule):
    """
    This child class represents all cyclic SubMolecules.
    The atoms must be in ring of size 3 to 8.
    """
    COLOR = (0, 1, 0)

    def __init__(self, mol: Mol, hit_atoms: tuple[int], hit_str: str):
        super().__init__(mol, hit_atoms, hit_str)

    @staticmethod
    def submolecule_decider(mol: Mol, hits: tuple[int]) -> list[int]:
        """
        This function verifies that the atoms are in a ring
        """
        for atom in [mol.GetAtomWithIdx(hit) for hit in hits]:
            if not atom.IsInRing():
                return None

        return list(hits)

    @classmethod
    def get_hit_str(cls) -> list[str]:
        # Rings size 6 are the max found in the lipid maps database
        return ["[R]1@[R]@[R]1",  # 3-Ring
                "[R]1@[R]@[R]@[R]1",  # 4-Ring
                "[R]1@[R]@[R]@[R]@[R]1",  # 5-Ring
                "[R]1@[R]@[R]@[R]@[R]@[R]1",  # 6-Ring
                "[R]1@[R]@[R]@[R]@[R]@[R]@[R]1",  # 7-Ring
                "[R]1@[R]@[R]@[R]@[R]@[R]@[R]@[R]1",  # 8-Ring
                ]

    def _generate_bonds_list(self):
        for idx in range(len(self._atoms) - 1):
            bond = self._mol.GetBondBetweenAtoms(self._atoms[idx], self._atoms[idx + 1])
            self._bonds.append(bond.GetIdx())

        # close the ring
        bond = self._mol.GetBondBetweenAtoms(self._atoms[-1], self._atoms[0])
        self._bonds.append(bond.GetIdx())

    def optimize(self):
        """
        Atoms are allowed to be in multiple CyclicGroup, as long as they aren't identical
        """
        num_of_atoms = len(self.get_atoms())
        for atom in [self._mol.GetAtomWithIdx(atom_idx) for atom_idx in self.get_atoms()]:
            if atom.HasProp(self.get_type()):
                num_of_atoms -= 1

        # True if all atoms are assigned
        if bool(num_of_atoms):
            return self.__class__
        return None

    def sort_connections(self, edge_size_dict: dict[int, list[(int,)]]):
        """
        This function sorts based on the size of the connection_point for the first entry.
        The second entry is the closest then largest connection_point.
        Afterward it follows the direction.
        """

        # Helper Function to calculate size at connection point
        def sizeof(edge_size_dict_entry):
            return sum([entry[0] for entry in edge_size_dict_entry])

        con_points = self.get_con_points()
        con_len = len(con_points)
        _atoms = self.get_atoms()
        ring_size = len(_atoms)
        distances_to_next = []
        max_size, max_size_idxs = -1, []

        # Creating a list of distances to the next connection point.
        for idx in range(con_len):
            # calculate distance
            con_point_current = con_points[idx]
            con_point_next = con_points[(idx + 1) % len(con_points)]
            distance = (_atoms.index(con_point_next) - _atoms.index(con_point_current)) % ring_size
            distances_to_next.append(distance)

            # safe indices of largest con_points
            size = sizeof(edge_size_dict[con_point_current])
            if size >= max_size:
                if size == max_size:
                    max_size_idxs.append(idx)
                else:
                    max_size_idxs = [idx]
                max_size = size

        # Finding the con_point with the largest distance to next.
        # This point has the smallest sum of distances, when moving in the opposite direction.
        max_dist, cut_idx = 0, -1
        change_dir = False
        for idx in max_size_idxs:
            prev_idx = (idx - 1) % len(distances_to_next)
            if max_dist < distances_to_next[idx]:
                max_dist, cut_idx, change_dir = distances_to_next[idx], idx, True
            if max_dist < distances_to_next[idx]:
                max_dist, cut_idx, change_dir = distances_to_next[prev_idx], idx, False

        # modifying con_points, so it has the sorted order
        if change_dir:
            con_points.reverse()
            cut_idx = (len(con_points) - 1) - cut_idx

        # Moving largest con_point to start of array without changing order
        con_points = con_points[cut_idx:] + con_points[:cut_idx]

        # generate sorted_edges based on the order of con_points
        sorted_edges = []
        for con_point in con_points:
            # sorts edges on same con_point
            edge_size_dict[con_point].sort(key=self._sort_connections_helper, reverse=True)
            for _, edge in edge_size_dict[con_point]:
                sorted_edges.append(edge)
        return sorted_edges

    def get_con_point_metric(self):
        mean_distance = 0
        if len(self.get_con_points()) < 2:
            # there is no distance between less than 2 connection points
            return mean_distance

        # the metric is the square root of the distances squared
        prev_idx = None
        for cp in self.get_con_points():
            if prev_idx is not None:
                current_idx = self.get_atoms().index(cp)
                mean_distance += pow(current_idx - prev_idx, 2)
                prev_idx = current_idx
            else:
                prev_idx = self.get_atoms().index(cp)

        return math.sqrt(mean_distance / len(self.get_con_points()))
