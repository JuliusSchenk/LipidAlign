import math
from copy import deepcopy
from io import BytesIO
from random import random

import numpy as np
from PIL import Image as pilImage
from rdkit.Chem.Draw import MolDraw2DCairo

from src.molecule.graphed_molecule import GraphedMolecule
from src.molecule.sub_molecule import CyclicGroup, AliphaticChain, FunctionalGroup


class Alignment:
    def __init__(self, graphed_mol_0: GraphedMolecule, graphed_mol_1: GraphedMolecule, imagename="alignment"):
        self.graphs = [graphed_mol_0, graphed_mol_1]
        self.imagename = str(imagename)
        self.fps = None
        self.config = []
        # 0 const, 1 atoms_weight, 2 carbon_weight, 3 cp_weight, 4 cp_metric_weight, 5 type_weight, 6 base_weight, 7 expontent, 8 fps_radius, 9 alignmentstarts
        # TODO: Edit these values to "customize" your Alignment parameters.
        # For each Row there is Alignment
        self.config_list = [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.75, 1.0, 1, 10],
            [1.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.75, 1.0, 1, 10],
            [1.0, 0.0, 0.0, 10.0, 10.0, 0.0, 0.75, 10.0, 10, 10],
            # [1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 0.75, 1.0, 2, 10],
            # [1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 0.75, 1.0, 2, 10],
            # [1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 0.75, 1.0, 2, 10],
            # [1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 0.75, 1.0, 2, 10],
            # [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.75, 10.0, 2, 10],
        ]
        self._vec_buffer = {0: [[]]}
        self.draw()

    @staticmethod
    def get_highlights(alignments, rnd_col=False):
        align_col = [(1, 0, 0), (0, 1, 1), (1, 0.5, 0), (0, 0.5, 1), (1, 1, 0), (0, 0, 1),
                     (0.5, 1, 0), (0.5, 0, 1), (0, 1, 0), (1, 0, 1), (0, 1, 0.5), (1, 0, 0.5)]
        col_idx = 0
        atom_highlights = [{}, {}]
        bond_highlights = [{}, {}]
        for align_idx, sms in enumerate(alignments):
            if rnd_col:
                color = (random(), random(), random())
            else:
                if None in sms:
                    color = (0, 0, 0)
                else:
                    color = align_col[col_idx % len(align_col)]
                    col_idx += 1
            for idx, sm in enumerate(sms):
                if sm is not None:
                    atoms = sm.get_atoms()
                    for atom in atoms:
                        if atom in atom_highlights[idx]:
                            atom_highlights[idx][atom].append(color)
                        else:
                            atom_highlights[idx][atom] = [color]

                    bonds = sm.get_bonds()
                    for bond in bonds:
                        if bond in bond_highlights[idx]:
                            bond_highlights[idx][bond].append(color)
                        else:
                            bond_highlights[idx][bond] = [color]
        return atom_highlights, bond_highlights

    def draw(self, path="images/", grid_size=(750, 750)):
        # Implementation based on this blog entry:
        # http://rdkit.blogspot.com/2020/10/molecule-highlighting-and-r-group.html

        mol_pngs = []
        mols = [graph.get_mol() for graph in self.graphs]
        for idx_w in range(len(self.config_list)):
            self.config = self.config_list[idx_w]
            alignments = self.align_submols()
            atom_highlights_list, bond_highlights_list = self.get_highlights(alignments)

            # Generating subimages
            for i, mol in enumerate(mols):
                mol_pngs.append(
                    self.draw_mol(mol, atom_highlights=atom_highlights_list[i], bond_highlights=bond_highlights_list[i],
                                  indexing=True, size=grid_size))

        # Merging images
        cols, rows = 2, len(self.config_list)
        image_size = (grid_size[0] * cols, grid_size[1] * rows)
        align_png = pilImage.new('RGB', image_size)
        for idx_png, mol_png in enumerate(mol_pngs):
            png_bytes = BytesIO(mol_png)
            align_png.paste(pilImage.open(png_bytes),
                            box=(grid_size[0] * (idx_png % cols), grid_size[1] * (idx_png // 2)))

        path = path + self.imagename + ".png"
        print(f"Alignment saved at \'{path}\'")
        align_png.save(path, format='PNG')

    def draw_mol(self, mol, atom_highlights, bond_highlights, indexing=False,
                 size=(500, 500)):
        d = MolDraw2DCairo(size[0], size[1])
        d.drawOptions().addStereoAnnotation = True
        d.drawOptions().addAtomIndices = indexing
        d.drawOptions().fillHighlights = False
        radius, line_width = {}, {}
        legend = mol.GetProp("_Name") + "\n" + \
                 "Weights : {" + f"const:{self.config[0]}, atoms_weight:{self.config[1]}, carbon_weight:{self.config[2]},\n" \
                                 f"cp_weight:{self.config[3]}, cp_metric_weight:{self.config[4]}, type_weight:{self.config[5]},\n" \
                                 f"base_weight:{self.config[6]}, expontent:{self.config[7]}, fp_radius:{self.config[8]}, " + "}"
        d.DrawMoleculeWithHighlights(mol, legend=legend,
                                     highlight_atom_map=atom_highlights,
                                     highlight_bond_map=bond_highlights, highlight_radii=radius,
                                     highlight_linewidth_multipliers=line_width)
        d.FinishDrawing()
        return d.GetDrawingText()

    # these values are an estimated guess, difference is a measure of similarity
    SIMILARITY_MAP = {
        CyclicGroup: 1.0,
        AliphaticChain: 0.50,
        FunctionalGroup: 0.0,
    }

    def _get_fingerprints(self, radius=1):
        radius = max(radius, 1)
        # weights for the fingerprint of a submol
        const, atoms_weight, carbon_weight, cp_weight, cp_metric_weight, type_weight, base_weight = self.config[:7]
        # weight of each submol when higher radius
        neigh_weight = 1 - base_weight

        identifier_map = [{}, {}]
        # Generate fingerprints for each
        for idx, mol_graph in enumerate(self.graphs):
            pre_fingerprints = {}
            included_sms_map = {}
            submols = mol_graph.get_mol_class().get_sub_molecules()

            # First iteration needs special conditions and initialization
            for sm in submols:
                included_sms_map[sm] = [[sm]]
                fingerprint = [const,  # constant to get rough size after normalizing this vector
                               len(sm.get_atoms()) * atoms_weight,
                               # total number of non-carbons in submolecule
                               (len(sm.get_atoms()) - sm.get_num_of_carbons()) * carbon_weight,
                               # total number of carbons in submolecule
                               len(sm.get_con_points()) * cp_weight,  # number of connected submolecules
                               sm.get_con_point_metric() * cp_metric_weight,  # Somehow include neighbors as a value
                               self.SIMILARITY_MAP[type(sm)] * type_weight,
                               ]
                pre_fingerprints[sm] = [np.array(fingerprint, dtype=float, copy=False)]
                identifier_map[idx][sm] = []

            # the first radius is done by loop above
            for r in range(radius - 1):
                # With higher radii, the fingerprint of a submol includes more neighboring submols
                for sm in submols:
                    included_sms = []
                    np_fingerprint = np.array([0, 0, 0, 0, 0, 0], dtype=float, copy=False)
                    # get all neighboring submols
                    for outer_sm in included_sms_map[sm][r]:
                        for v_edges in mol_graph.get_graph()[outer_sm].get_edges():
                            neigh_sm = v_edges.get_other_sm(outer_sm)
                            included_sms.append(neigh_sm)
                            np_fingerprint += pre_fingerprints[neigh_sm][0]
                        # remove duplicates and sm from a lower radius
                        included_sms = list(set(included_sms) - set(included_sms_map[sm][r]))
                        included_sms_map[sm].append(included_sms)

                        # we want that first vector entry is always 1
                        np_fingerprint = pre_fingerprints[sm][r] * base_weight + np_fingerprint * neigh_weight
                        pre_fingerprints[sm].append(np_fingerprint)

            # normalize the result and save them
            for sm, np_fingerprints in pre_fingerprints.items():
                identifier = []
                for np_fingerprint in np_fingerprints:
                    # normalizing doesn't really make sense in this case
                    np_fingerprint_norm = np_fingerprint / np.linalg.norm(np_fingerprint) * 10
                    identifier.append(np_fingerprint_norm)
                identifier_map[idx][sm] = identifier

        return identifier_map

    def align_submols(self):
        fps_radius = self.config[8]
        self.fps = self._get_fingerprints(fps_radius)
        max_seepoints = self.config[9]

        # Get Submols with the most similarity
        seedpoints = []

        for sm_0 in self.graphs[0].get_mol_class().get_sub_molecules():
            for sm_1 in self.graphs[1].get_mol_class().get_sub_molecules():
                # if sm_0.get_type() != sm_0.get_type():
                if sm_0.get_name() != sm_0.get_name():
                    continue
                cost = self.get_cost(_sm_0=sm_0, _sm_1=sm_1, idx_0=-1, idx_1=-1)
                for seed_idx, seedpoint in enumerate(seedpoints):
                    if seedpoint[2] > cost:
                        seedpoints.insert(seed_idx, (sm_0, sm_1, cost))
                        break
                    elif seed_idx > max_seepoints:
                        seedpoints.append((sm_0, sm_1, cost))
                        break
                else:
                    # the cost is the highest yet
                    seedpoints.append((sm_0, sm_1, cost))

        cost = math.inf
        alignment = []
        seed_point_amount = min(max_seepoints, len(seedpoints))
        for idx in range(seed_point_amount):
            sm_0 = seedpoints[idx][0]
            sm_1 = seedpoints[idx][1]
            tmp_cost, tmp_alignment = self._align_submol_helper_depthfirst(sm_0=sm_0, sm_1=sm_1)

            if cost > tmp_cost:
                cost = tmp_cost
                alignment = tmp_alignment.copy()

        return alignment  # list of tuples

    def get_cost(self, _sm_0, _sm_1, idx_0=0, idx_1=0):
        cost_exponent = self.config[7]
        if _sm_0 is None and _sm_1 is None:
            return 0
        elif _sm_0 is None:
            return np.linalg.norm(np.power((self.fps[1][_sm_1][idx_1]), cost_exponent))
        elif _sm_1 is None:
            return np.linalg.norm(np.power((self.fps[0][_sm_0][idx_0]), cost_exponent))

        return np.linalg.norm(np.power(self.fps[0][_sm_0][idx_0] - self.fps[1][_sm_1][idx_1], cost_exponent))

    def _align_submol_helper_depthfirst(self, sm_0, sm_1):
        """
        This function uses depth-first search to aligns the two molecules based on a fixed alignment start
        :param sm_0: submolecule 0
        :param sm_1: submolecule 1
        :return: cost of the alignment and the alignment
        """
        sm_0_neighbors = [e.get_other_sm(sm_0) for e in self.graphs[0].get_graph()[sm_0].get_edges()]
        sm_1_neighbors = [e.get_other_sm(sm_1) for e in self.graphs[1].get_graph()[sm_1].get_edges()]
        diff = len(sm_0_neighbors) - len(sm_1_neighbors)
        if diff < 0:
            sm_0_neighbors += [None] * -diff
        elif diff > 0:
            sm_1_neighbors += [None] * diff
        align_vec = self._align_vec(len(sm_1_neighbors))
        min_cost = math.inf
        min_alignment = []
        for reverse_list in [False, True]:
            # it is needed 2 reverse one of the list to have every possibility
            if reverse_list:
                sm_0_neighbors.reverse()
                num_of_none = 0
                while None in sm_0_neighbors:
                    num_of_none += 1
                    sm_0_neighbors.remove(None)
                sm_0_neighbors += [None] * num_of_none

            for align_v in align_vec:
                for swap in [False, True]:
                    sm_0_neigh_idx, sm_1_neigh_idx = 0, 0
                    tmp_cost = 0.0
                    tmp_align = []
                    tmp_already_visit_extras = []
                    for match in align_v:
                        if match:
                            # Align Submols at given index
                            new_sm_0 = sm_0_neighbors[sm_0_neigh_idx]
                            new_sm_1 = sm_1_neighbors[sm_1_neigh_idx]
                            if new_sm_0 and new_sm_1:
                                new_cost, new_align, new_already_visited = \
                                    self._align_submol_helper_depthfirst_rec(new_sm_0, new_sm_1, (sm_0, sm_1),
                                                                             ([sm_0, sm_1] + tmp_already_visit_extras))
                            elif new_sm_0:
                                new_cost, new_align, new_already_visited = \
                                    self._delete_submol_downstream(new_sm_0, 0, (sm_0, sm_1),
                                                                   ([sm_0, sm_1] + tmp_already_visit_extras))
                            elif new_sm_1:
                                new_cost, new_align, new_already_visited = \
                                    self._delete_submol_downstream(new_sm_1, 1, (sm_0, sm_1),
                                                                   ([sm_0, sm_1] + tmp_already_visit_extras))
                            else:
                                raise ValueError
                            tmp_cost += new_cost
                            tmp_align += new_align
                            tmp_already_visit_extras += new_already_visited
                            sm_0_neigh_idx += 1
                            sm_1_neigh_idx += 1
                        elif swap:
                            # align sm_o with None
                            new_sm_0 = sm_0_neighbors[sm_0_neigh_idx]
                            new_cost, new_align, new_already_visited = \
                                self._delete_submol_downstream(new_sm_0, 0, (sm_0, sm_1),
                                                               [sm_0, sm_1] + tmp_already_visit_extras)
                            tmp_cost += new_cost
                            tmp_align += new_align
                            tmp_already_visit_extras += new_already_visited
                            sm_0_neigh_idx += 1
                        else:
                            # align sm_1 with None
                            new_sm_1 = sm_1_neighbors[sm_1_neigh_idx]
                            new_cost, new_align, new_already_visited = \
                                self._delete_submol_downstream(new_sm_1, 1, (sm_0, sm_1),
                                                               [sm_0, sm_1] + tmp_already_visit_extras)
                            tmp_cost += new_cost
                            tmp_align += new_align
                            tmp_already_visit_extras += new_already_visited
                            sm_1_neigh_idx += 1

                    # Add cost of non submols aligned with None
                    for sm_0_rest in range(sm_0_neigh_idx, len(sm_0_neighbors)):
                        sm_0_neigh = sm_0_neighbors[sm_0_rest]
                        if sm_0_neigh:
                            new_cost, new_align, _ = \
                                self._delete_submol_downstream(sm_0_neigh, 0, (sm_0, sm_1),
                                                               [sm_0, sm_1] + tmp_already_visit_extras)
                            tmp_cost += new_cost
                            tmp_align += new_align
                    for sm_1_rest in range(sm_1_neigh_idx, len(sm_1_neighbors)):
                        sm_1_neigh = sm_1_neighbors[sm_1_rest]
                        if sm_1_neigh:
                            new_cost, new_align, _ = \
                                self._delete_submol_downstream(sm_1_neigh, 1, (sm_0, sm_1),
                                                               [sm_0, sm_1] + tmp_already_visit_extras)
                            tmp_cost += new_cost
                            tmp_align += new_align

                    if min_cost > tmp_cost:
                        min_cost = tmp_cost
                        min_alignment = tmp_align

        alignment = min_alignment
        cost = min_cost
        alignment.append((sm_0, sm_1))
        cost += self.get_cost(sm_0, sm_1)
        return cost, alignment

    def _align_submol_helper_depthfirst_rec(self, sm_0, sm_1, source, already_visited):
        _already_visited = []
        if sm_0 is None and sm_1 is None:
            return 0.0, [], []
        sub_align = []
        sub_cost = 0.0
        sm_0_neighbors = []
        sm_1_neighbors = []
        if sm_0:
            sm_0_neighbors = [e.get_other_sm(sm_0) for e in self.graphs[0].get_graph()[sm_0].get_edges()]
        if sm_1:
            sm_1_neighbors = [e.get_other_sm(sm_1) for e in self.graphs[1].get_graph()[sm_1].get_edges()]

        # mark already aligned Submols
        already_aligned_idx = [-1, -1]
        for sm in sm_0_neighbors:
            if sm is source[0] and sm:
                already_aligned_idx[0] = (sm_0_neighbors.index(sm))
            elif sm in already_visited:
                sm_0_neighbors.remove(sm)
        for sm in sm_1_neighbors:
            if sm is source[1] and sm:
                already_aligned_idx[1] = sm_1_neighbors.index(sm)
            elif sm in already_visited:
                sm_1_neighbors.remove(sm)

        # Nothing more to align
        if len(sm_0_neighbors) == 0 and len(sm_1_neighbors) == 0:
            sub_align.append((sm_0, sm_1))
            sub_cost += self.get_cost(sm_0, sm_1)
            _already_visited.append(sm_0)
            _already_visited.append(sm_1)
            return sub_cost, sub_align, _already_visited
        elif len(sm_0_neighbors) == 0 or len(sm_1_neighbors) == 0:
            if len(sm_0_neighbors) == 0:
                for sm_1_s in sm_1_neighbors:
                    if not (sm_1_s is sm_1):
                        new_cost, new_align, new_already_visited = \
                            self._delete_submol_downstream(sm_1_s, 1, (sm_0, sm_1), already_visited)
                        sub_cost += new_cost
                        sub_align += new_align
                        _already_visited += new_already_visited
                return sub_cost, sub_align, _already_visited

            if len(sm_1_neighbors) == 0:
                for sm_0_s in sm_0_neighbors:
                    if sm_0_s is sm_0:
                        new_cost, new_align, new_already_visited = \
                            self._delete_submol_downstream(sm_0_s, 0, (sm_0, sm_1), sm_0_neighbors + already_visited)
                        sub_cost += new_cost
                        sub_align += new_align
                        print(new_align)
                        _already_visited += new_already_visited
                return sub_cost, sub_align, _already_visited
        else:
            if -1 in already_aligned_idx:
                if already_aligned_idx[0] != -1:
                    sm_0_neighbors.pop(already_aligned_idx[0])
                if already_aligned_idx[1] != -1:
                    sm_1_neighbors.pop(already_aligned_idx[1])
                diff = len(sm_0_neighbors) - len(sm_1_neighbors)
                if diff < 0:
                    sm_0_neighbors += [None] * -diff
                elif diff > 0:
                    sm_1_neighbors += [None] * diff

                sm_0_neigh_splits = [sm_0_neighbors]
                sm_1_neigh_splits = [sm_1_neighbors]
            else:
                sm_0_neigh_split_first = sm_0_neighbors[:already_aligned_idx[0]]
                sm_0_neigh_split_last = sm_0_neighbors[already_aligned_idx[0] + 1:]

                sm_1_neigh_split_first = sm_1_neighbors[:already_aligned_idx[1]]
                sm_1_neigh_split_last = sm_1_neighbors[already_aligned_idx[1] + 1:]

                # Fill list with Nones so they are the same length
                diff = len(sm_0_neigh_split_first) - len(sm_1_neigh_split_first)
                if diff < 0:
                    sm_0_neigh_split_first += [None] * -diff
                elif diff > 0:
                    sm_1_neigh_split_first += [None] * diff

                diff = len(sm_0_neigh_split_last) - len(sm_1_neigh_split_last)
                if diff < 0:
                    sm_0_neigh_split_last += [None] * -diff
                elif diff > 0:
                    sm_1_neigh_split_last += [None] * diff

                # reverse the first list, so the first Submol is closes to the connection point
                sm_0_neigh_split_first.reverse()
                sm_1_neigh_split_first.reverse()
                sm_0_neigh_splits = [sm_0_neigh_split_first] + [sm_0_neigh_split_last]
                sm_1_neigh_splits = [sm_1_neigh_split_first] + [sm_1_neigh_split_last]

            r_sub_align_total = []
            r_already_visited_total = []
            r_sub_cost_total = math.inf
            for reverse_list in [False, True]:
                r_sub_align = []
                r_already_visited = []
                r_sub_cost = 0

                if reverse_list:
                    if len(sm_0_neigh_splits) > 1:
                        # remove trailing Nones
                        for sm_0_neigh_split in sm_0_neigh_splits:
                            while None in sm_0_neigh_split:
                                sm_0_neigh_split.remove(None)
                        for sm_1_neigh_split in sm_1_neigh_splits:
                            while None in sm_1_neigh_split:
                                sm_1_neigh_split.remove(None)

                        # Reversing sm_0_list
                        sm_0_neigh_splits.reverse()

                        # reverse to add trailing Nones
                        sm_0_neigh_splits[1].reverse()
                        sm_1_neigh_splits[0].reverse()

                        # Fill list with Nones so they are the same length
                        for sm_0_neigh_split, sm_1_neigh_split in zip(sm_0_neigh_splits, sm_1_neigh_splits):

                            diff = len(sm_0_neigh_split) - len(sm_1_neigh_split)
                            if diff < 0:
                                sm_0_neigh_split += [None] * -diff
                            elif diff > 0:
                                sm_1_neigh_split += [None] * diff

                        # reverse for right order
                        sm_0_neigh_splits[0].reverse()
                        sm_1_neigh_splits[0].reverse()
                    else:
                        break

                # Generate the alignment patterns
                align_vecs = []
                for split in sm_0_neigh_splits:
                    align_vecs.append(self._align_vec(len(split)))

                # Try every viable combination recursively till there is a trivial solution.
                for align_vec, sm_0_neigh_split, sm_1_neigh_split in zip(align_vecs, sm_0_neigh_splits,
                                                                         sm_1_neigh_splits):
                    split_align = []
                    split_already_visit_extras = []
                    split_cost = math.inf
                    # align_vec is a list of matching patterns, try every and return the cheapest one
                    for align_v in align_vec:
                        if not align_v:
                            continue

                        swap_cost = math.inf
                        swap_align = []
                        swap_already_visit_extras = []
                        for swap in [False, True]:
                            sm_0_neigh_idx, sm_1_neigh_idx = 0, 0
                            tmp_cost = 0.0
                            tmp_align = []
                            tmp_already_visit_extras = []
                            for match in align_v:
                                if match:
                                    # Align Submols at given index
                                    new_sm_0 = sm_0_neigh_split[sm_0_neigh_idx]
                                    new_sm_1 = sm_1_neigh_split[sm_1_neigh_idx]

                                    if new_sm_0 and new_sm_1:
                                        new_cost, new_align, new_already_visited = \
                                            self._align_submol_helper_depthfirst_rec(new_sm_0, new_sm_1, (sm_0, sm_1), (
                                                    already_visited + tmp_already_visit_extras))
                                    elif new_sm_0:
                                        new_cost, new_align, new_already_visited = \
                                            self._delete_submol_downstream(new_sm_0, 0, (sm_0, sm_1),
                                                                           (already_visited + tmp_already_visit_extras))
                                    elif new_sm_1:
                                        new_cost, new_align, new_already_visited = \
                                            self._delete_submol_downstream(new_sm_1, 1, (sm_0, sm_1),
                                                                           (already_visited + tmp_already_visit_extras))
                                    else:
                                        raise ValueError

                                    tmp_already_visit_extras += new_already_visited
                                    tmp_cost += new_cost
                                    tmp_align += new_align
                                    sm_0_neigh_idx += 1
                                    sm_1_neigh_idx += 1
                                elif swap:
                                    # align sm_o with None
                                    new_sm_0 = sm_0_neigh_split[sm_0_neigh_idx]
                                    new_cost, new_align, new_already_visited = \
                                        self._delete_submol_downstream(new_sm_0, 0, (sm_0, sm_1),
                                                                       (already_visited + tmp_already_visit_extras))
                                    tmp_already_visit_extras += new_already_visited
                                    tmp_cost += new_cost
                                    tmp_align += new_align
                                    sm_0_neigh_idx += 1
                                else:
                                    # align sm_1 with None
                                    new_sm_1 = sm_1_neigh_split[sm_1_neigh_idx]
                                    new_cost, new_align, new_already_visited = \
                                        self._delete_submol_downstream(new_sm_1, 1, (sm_0, sm_1),
                                                                       (already_visited + tmp_already_visit_extras))
                                    tmp_already_visit_extras += new_already_visited
                                    tmp_cost += new_cost
                                    tmp_align += new_align
                                    sm_1_neigh_idx += 1
                            # Add cost of non submols aligned with None
                            for sm_0_rest in range(sm_0_neigh_idx, len(sm_0_neigh_split)):
                                sm_0_neigh = sm_0_neigh_split[sm_0_rest]
                                if sm_0_neigh:
                                    new_cost, new_align, new_already_visited = \
                                        self._delete_submol_downstream(sm_0_neigh, 0, (sm_0, sm_1),
                                                                       (already_visited + tmp_already_visit_extras))
                                    tmp_cost += new_cost
                                    tmp_align += new_align
                                    tmp_already_visit_extras += new_already_visited
                            for sm_1_rest in range(sm_1_neigh_idx, len(sm_1_neigh_split)):
                                sm_1_neigh = sm_1_neigh_split[sm_1_rest]
                                if sm_1_neigh:
                                    new_cost, new_align, new_already_visited = \
                                        self._delete_submol_downstream(sm_1_neigh, 1, (sm_0, sm_1),
                                                                       (already_visited + tmp_already_visit_extras))
                                    tmp_cost += new_cost
                                    tmp_align += new_align
                                    tmp_already_visit_extras += new_already_visited

                            # this alignment is currently the best for this swap
                            if swap_cost > tmp_cost:
                                swap_cost = tmp_cost
                                swap_align = tmp_align
                                swap_already_visit_extras = tmp_already_visit_extras

                        # this alignment is currently the best for this split
                        if split_cost > swap_cost:
                            split_cost = swap_cost
                            split_align = swap_align
                            split_already_visit_extras = swap_already_visit_extras

                    # this alignment is the best for this split
                    if split_cost != math.inf:
                        r_sub_cost += split_cost
                        r_sub_align += split_align
                        r_already_visited += split_already_visit_extras

                # this alignment is currently the best for this Submol
                if r_sub_cost_total > r_sub_cost:
                    r_sub_cost_total = r_sub_cost
                    r_sub_align_total = r_sub_align
                    r_already_visited_total = r_already_visited

            # this alignment is the best for this Submol
            if r_sub_cost_total != math.inf:
                sub_align += r_sub_align_total
                sub_cost += r_sub_cost_total
                _already_visited += r_already_visited_total

        sub_cost += self.get_cost(sm_0, sm_1)
        sub_align.append((sm_0, sm_1))
        return sub_cost, sub_align, _already_visited

    def _delete_submol_downstream(self, sm, mol_idx, source, already_visited):
        _already_visited = []
        if sm is None:
            return 0.0, [], []

        # Get all downstream submols
        edges = self.graphs[mol_idx].get_graph()[source[mol_idx]].get_edges()
        edge = None
        for possible_edge in edges:
            if possible_edge.get_other_sm(source[mol_idx]) is sm:
                edge = possible_edge
        sms_downstream = [node.get_submol() for node in
                          self.graphs[mol_idx].get_graph()[source[mol_idx]].get_downstream_nodes(edge,
                                                                                                 invert_edge=True)]

        cost = 0.0
        alignment = []
        if mol_idx == 0:
            for sm_ds in sms_downstream:
                if sm_ds in already_visited:
                    continue
                cost += self.get_cost(sm_ds, None)
                alignment.append((sm_ds, None))
                _already_visited.append(sm_ds)
        elif mol_idx == 1:
            for sm_ds in sms_downstream:
                if sm_ds in already_visited:
                    continue
                cost += self.get_cost(None, sm_ds)
                alignment.append((None, sm_ds))
                _already_visited.append(sm_ds)

        return cost, alignment, _already_visited

    def _align_vec(self, n):
        """
        This function generates all possible permutations of size n with symbolset = [True,False]
        Because this function will often be called with similar n, we can buffer them for faster excess.
        :param n: size of the vector
        :return: All possible permutations
        """
        if n < 0:
            return self._vec_buffer[0]

        if len(self._vec_buffer) > n:
            return self._vec_buffer[n]

        for idx in range(len(self._vec_buffer) - 1, n):
            align_vs = deepcopy(self._vec_buffer[idx])
            align_vs_len = len(align_vs)
            for v_idx in range(align_vs_len):
                v_clone = deepcopy(align_vs[v_idx])
                v_clone.append(False)
                align_vs[v_idx].append(True)
                align_vs.append(v_clone)
            self._vec_buffer[idx + 1] = align_vs

        return self._vec_buffer[n]
