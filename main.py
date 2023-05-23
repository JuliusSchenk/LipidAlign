import random
import sys

from src.alignment import Alignment
from src.molecule import GraphedMolecule
from src.molecule import Molecule


def get_fatty_acryls():
    return [
        # fatty acids
        Molecule("O=C(O)CCCCCCCCCCCCCCCCCCCCCCCCC", "Cerotic acid"),
        Molecule("CCCCCCCC\\C=C/CCCCCCCC(O)=O", "Oleic acid"),
        Molecule("O=C(O)CCCCCCC/C=C/CCCCCCCC", "Elaidic acid"),
        Molecule("O=C(O)CCC(CCCCC(CCC)C)CCCC", "unknown branched f. acid"),
        # fatty alcohol
        Molecule("OCCCCCCCCCCCC", "Dodecanol"),
        Molecule("OCCCCCCCCCCCCCCCCCCCC", "Eicosanol"),
    ]


def get_glycerolipids():
    return [
        Molecule("CCCCCCCCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCCCCCCCC)OC(=O)CCCCCCCCCCCCCCCCC", "Stearin"),
    ]


def get_glycerophospholipids():
    return [
        # Glycerophospholipids
        Molecule("CCCCCCCC\\C=C/CCCCCCCC(=O)OCC(COP(=O)(O)O)O", "Lysophosphatidic acid"),
        Molecule("O=C(OC[C@@H](OC(=O)CCCCCCCCCCCCCCC)COP([O-])(=O)OCC[N+](C)(C)C)CCCCCCCCCCCCCCC",
                 "Dipalmitoylphosphatidylcholine"),
        Molecule(
            "CCCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)(O)OC1C([C@@H](C(C(C1O)O)OP(=O)(O)O)O)O)OC(=O)CCC/C=C\C/C=C\C/C=C\C/C=C\CCCCC",
            "Phosphatidylinositol-4-phosphate"),
    ]


def get_sphingolipids():
    return [
        # Sphingolipids
        Molecule("CCCCCCCCCCCCC/C=C/[C@H]([C@H](CO)N)O", "Sphingosine"),
        Molecule("CCCCCCCCCCCCC/C=C/[C@H]([C@H](COP(=O)([O-])OCC[N+](C)(C)C)NC(=O)CCCCCCCCCCCCC)O",
                 "C14 Sphingomyelin"),
    ]


def get_sterol_lipids():
    return [
        # Sterols
        Molecule("C1CC2CCC3C4CCC(CC4CCC3C2C1)O", "Sterol"),
        Molecule("C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C", "Cholesterol"),
    ]


def get_prenol_lipids():
    return [
        # Prenols
        Molecule("OC\\C=C(/C)C", "Prenol"),
        Molecule("CC(=CCC/C(=C/CO)/C)C", "Geraniol"),
        Molecule("OC\\C=C(/CC/C=C(/C)C)C", "Nerol"),
        Molecule("OC\\C=C(/C)CC\\C=C(/C)CC\\C=C(/C)CC\\C=C(/C)CC\\C=C(/C)CC\\C=C(/C)C", "Polyprenol"),
    ]


def get_saccharolipids():
    return [
        # Saccharolipids
        Molecule(
            "CCCCCCCCCCCCCC(=O)O[C@H](CCCCCCCCCCC)CC(=O)O[C@@H]1[C@H]([C@@H](O[C@@H]([C@H]1OP(=O)(O)O)CO)OC[C@@H]2[C@H]([C@@H]([C@H]([C@H](O2)OP(=O)(O)O)NC(=O)C[C@@H](CCCCCCCCCCC)O)OC(=O)C[C@@H](CCCCCCCCCCC)O)O)NC(=O)C[C@@H](CCCCCCCCCCC)OC(=O)CCCCCCCCCCC",
            "Lipid A"),
    ]


def get_polyketides():
    return [

    ]


def get_edgecases():
    return [
        # Molecule("CCCOC(=O)CCC", "unknown molecule"),
        Molecule("OCCCCCCCCCCCCCCCO", "Edgecase1"),
        Molecule("OCCCCCCCC(CCCCCCO)C", "Edgecase2"),
    ]


def get_cyclic_edgecases():
    return [
        Molecule("C1CC1", "Cyclopropane"),
        Molecule("c1cC1", "Cyclopropene"),
        Molecule("C1CCC1", "Cyclobutane"),
        Molecule("c1cCC1", "Cyclobutene"),
        Molecule("c1ccc1", "Cyclobuta-di-ene"),
        Molecule("C1CCCC1", "Cyclopentane"),
        Molecule("c1cCCC1", "Cyclopentene"),
        Molecule("c1cccC1", "Cyclopenta-di-ene"),
        Molecule("C1CCCCC1", "Cyclohexane"),
        Molecule("c1cCCCC1", "Cyclohexene"),
        Molecule("c1cccCC1", "1,3-Cyclohexa-di-ene"),
        Molecule("c1cCccC1", "1,4-Cyclohexa-di-ene"),
        Molecule("c1ccccc1", "Benzene"),
    ]


def get_test_molecules():
    return (
            get_fatty_acryls()
            + get_polyketides()
            + get_glycerolipids()
            + get_glycerophospholipids()
            + get_prenol_lipids()
            + get_sphingolipids()
            + get_saccharolipids()
            + get_sterol_lipids()
            + get_edgecases()
        #        + get_cyclic_edgecases()
    )


def get_alignment_mol():
    return [
        Molecule("O=C(O)CCC(C(CC)CCC)CCCCCC", "unknown branched f. acid"),
        Molecule("OCCCCCCCCCCCC", "Dodecanol"),
    ]


def molecule(mol_list):
    for mol in mol_list:
        mol.draw(indexing=True, coloring=True, random_coloring=False)


def get_random_mol(num, max_line=17777494):
    if not num:
        return []

    lines = []
    max_line = min(max_line, 17777494)  # length of the tsv file
    num = min(num, max_line)
    with open("src/testdata/lipid.tsv") as tsv:
        # readline and choose it with a chance
        next(tsv)  # skip header
        line_num = 1
        # Reservoir sampling - Algorithm with ability to fetch num-lines
        while num:
            temp_num = num
            tsv_line = None
            # tries num-times to determine if line is included
            for _ in range(temp_num):
                if random.random() < (line_num / (max_line - line_num)):
                    if not tsv_line:
                        tsv_line = tsv.readline()
                    lines.append(tsv_line)
                    num -= 1
            next(tsv)
            line_num += 1
    # we have lines, now we create the molecule-objects
    mols = []
    for line in lines:
        l = line.removesuffix("\n").split("\t")
        smile_str, name = l[4], l[2].replace(":", "")
        mols.append(Molecule(smile_str, name))

    return mols


def test():
    # This function tries to generate a Molecule object with every entry in th Lipidmaps tsv file
    with open("src/testdata/lipid.tsv") as tsv:
        next(tsv)
        for line in tsv:
            l = line.removesuffix("\n").split("\t")
            smile_str, name = l[4], l[2].replace(":", "")
            if (smile_str or smile_str == "[empty]") and name:
                try:
                    mol = Molecule(smile_str, name)

                except:
                    print(f"{name} failed with smile: {smile_str}")


def align(mol1: Molecule, mol2: Molecule, i):
    print(f"{mol1.get_name()} and {mol2.get_name()}")
    Alignment(GraphedMolecule(mol1),
              GraphedMolecule(mol2), i)


def rnd_align():
    num_of_alignments = 1
    if len(sys.argv) == 3:
        try:
            num_of_alignments = int(sys.argv[2])
        except:
            print(f"{sys.argv[2]} isn't a int.\nWill abort process")

    mols0 = get_random_mol(num_of_alignments)
    mols1 = get_random_mol(num_of_alignments)
    random.shuffle(mols0)
    random.shuffle(mols1)
    for idx, mols in enumerate(zip(mols0, mols1)):
        align(mols[0], mols[1], "alignment_" + str(idx))


def pre_align():
    mols0 = get_test_molecules()
    mols1 = get_test_molecules()
    random.shuffle(mols0)
    random.shuffle(mols1)

    for idx, mols in enumerate(zip(mols0, mols1)):
        align(mols[0], mols[1], "alignment_" + str(idx))


def usr_align():
    prompt_mol_0 = smiles_prompt("Please enter a SMILES-Str for the first Lipid:\n",
                                 input("Please enter a name for the first Lipid:\n"))
    prompt_mol_1 = smiles_prompt("Please enter a SMILES-Str for the second Lipid:\n",
                                 input("Please enter a name for the second Lipid:\n"))
    align(prompt_mol_0, prompt_mol_1, f"{prompt_mol_0.get_name()}_{prompt_mol_1.get_name()}_alignment")


def test_molecule():
    molecule([smiles_prompt("Please enter a SMILES-Str:\n", input("Please enter a name for the Lipid:\n"))])


def smiles_prompt(prompt, mol_name):
    smiles_string = input(prompt)
    smiles_string.strip()
    try:
        mol = Molecule(smiles_string, mol_name)
    except:
        print(f"Failed Lipid creation with given SMILES-Str: {smiles_string}")
        exit(1)

    return mol


flags = {
    "-r": rnd_align,
    "-R": pre_align,
    "-u": usr_align,
    "-t": test_molecule,
}


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] in flags:
            flags[sys.argv[1]]()
            return

    print("Usage: py ./main.py [flag]\n"
          "\t-r {n}\t: Fetches 2n Lipids and aligns them\n"
          "\t-R\t: Get a alignment image of predefined Lipids\n"
          "\t-u\t: User supplies the Lipids and a alignment image of it is created\n"
          "\t-t\t: User supplies the Lipid and a molecule image of it is created")


if __name__ == "__main__":
    main()
