from random import random

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from .sub_molecule import SubMolecule


class Molecule:
    """
    This Class represents a Molecule and implements all needed functions to work with them.
    """

    def __init__(self, smile_str, name='molecule'):
        self._sub_mols = None

        self._smile_str = smile_str
        self._name = name
        self._mol = Chem.RemoveHs(Chem.MolFromSmiles(self._smile_str))
        self._mol.SetProp("_Name", name)
        self.optimize_sub_molecules()

    def draw(self, path="images/", indexing=False, coloring=False, size=(1000, 1000),
             random_coloring=False):
        """
        This function creates a png of this molecule.
        :param path: path from main.py
        :param indexing: show atom indices
        :param size: png size
        :param coloring: Use color to show Submolecules
        :param random_coloring: Use random coloring for each Submolecule
        :return: None
        """
        highlight_atoms, highlight_bonds, color_atoms, color_bonds = [], [], {}, {}
        if coloring:
            for sm in self.get_sub_molecules():
                highlight_atoms = highlight_atoms + sm.get_atoms()
                highlight_bonds = highlight_bonds + sm.get_bonds()
                if not random_coloring:
                    sm_class = sm.__class__
                    color = sm_class.COLOR
                    color_atoms, color_bonds = sm.get_color(colors=(color_atoms, color_bonds), atom_col=color,
                                                            bond_col=color)
                else:
                    color = (random(), random(), random())
                    color_atoms, color_bonds = sm.get_color(colors=(color_atoms, color_bonds), atom_col=color,
                                                            bond_col=color)

        d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        d.drawOptions().addStereoAnnotation = True
        d.drawOptions().addAtomIndices = indexing
        d.drawOptions().fillHighlights = False
        radius, line_width = {}, {}
        d.DrawMoleculeWithHighlights(self._mol, self.get_name(), color_atoms, color_bonds, radius, line_width)
        d.FinishDrawing()
        path = path + self._name + '.png'
        print(f"Molecule image saved at \'{path}\'")
        d.WriteDrawingText(path)
        d.WriteDrawingText(path)

    @staticmethod
    def get_all_separators():
        """
        This function collects all separators/seed points used by Submolecule.
        :return: A list of all unique separators sorted by size
        """
        return SubMolecule.get_all_hit_str()

    def cut_mol(self, seps=None):
        """
        This function generates all matches for given seps (if not set use all).
        :param seps: A list of SMARTS strings to match on the molecules
        :return: A dictionary which maps separator with their matches, list of indices, in the molecule.
        """
        if seps is None:
            seps = self.get_all_separators()

        cut_dict = {sep: self._mol.GetSubstructMatches(Chem.MolFromSmarts(sep)) for sep in seps}

        return cut_dict

    def _generate_sub_mols(self):
        """
        This function generates all Submolecules for this molecule and saves them.
        These Submolecules aren't optimized, so there might be duplicates.
        :return: None
        """
        self._sub_mols = []
        for hit_str, hit_atoms_list in self.cut_mol(self.get_all_separators()).items():
            for hit_atoms in hit_atoms_list:
                # initiates Submolecule for given match. SubMolecule class decides which child class to use, if any.
                sm = SubMolecule(mol=self._mol, hit_atoms=hit_atoms, hit_str=hit_str)
                if sm is not None:
                    self._sub_mols.append(sm)

    def get_sub_molecules(self) -> list[SubMolecule]:
        """
        Returns all SubMolecules, if none are set then generates them.
        :return: A list of all SubMolecules
        """
        if self._sub_mols is None:
            self._generate_sub_mols()
        return self._sub_mols

    def set_sub_molecules(self, sub_mols: list[SubMolecule]):
        """
        Overwirtes the Submolecules
        :param sub_mols: A list SubMolecules
        :return:
        """
        self._sub_mols = sub_mols

    def get_mol(self):
        """
        :return: The rdkit Mol object
        """
        return self._mol

    def get_name(self):
        """
        :return: The name of the Molecule
        """
        return self._name

    def optimize_sub_molecules(self):
        """
        This function removes duplicates, optimizes the SubMolecules and sorts them based on the amount of atoms and type.
        :return: None
        """
        tmp_sub_mol_dict = {}
        sub_mols = self.get_sub_molecules()
        # sort by type
        for sub_mol in sub_mols:
            if sub_mol.get_type() not in tmp_sub_mol_dict:
                tmp_sub_mol_dict[sub_mol.get_type()] = []
            tmp_sub_mol_dict[sub_mol.get_type()].append(sub_mol)

        # reset for result
        sub_mols = []

        # sort each type by the list size
        for sub_mol_list in tmp_sub_mol_dict.values():
            sub_mol_list.sort(reverse=True)

        # remove duplicate assigned atoms
        for sub_mol_list in tmp_sub_mol_dict.values():
            for sub_mol in sub_mol_list:
                # Check if atoms aren't assigned already
                response_mol = sub_mol.optimize()
                # sub_mol is unique
                if response_mol:
                    sub_mols.append(sub_mol)
                    sub_mol.set_assignment()

        # overwrite old sub_mols
        self._sub_mols = sub_mols
