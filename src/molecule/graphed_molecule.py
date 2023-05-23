from .molecule import Molecule
from .sub_molecule import SubMolecule


class GraphedMolecule:
    """
    This class converts the Molecule class into a Graph, with nodes being the SubMolecules and the edges being the
    atoms, which connects them.
    """

    def __init__(self, mol_class: Molecule):
        self._mol_class = mol_class
        # convert Molecule to Submolecule graph
        self._graph = self.graphify(mol_class)

    def get_mol(self):
        return self._mol_class.get_mol()

    def get_graph(self):
        return self._graph

    def get_mol_class(self):
        return self._mol_class

    def graphify(self, mol_class: Molecule):
        """
        This function generates a graph and sorts the edges.
        :param mol_class: a Molecule object
        :return: A graph representing mol_class
        """
        # create Edges
        connections = self.get_connections()
        edges = []
        for atom_idx, sub_mols in connections.items():
            while len(sub_mols):
                sm = sub_mols.pop()
                for sub_mol in sub_mols:
                    edges.append(GraphedMolecule.Edge(atom_idx, sm, sub_mol))

        # create Nodes
        nodes_dict = {}
        for sm in mol_class.get_sub_molecules():
            nodes_dict[sm] = GraphedMolecule.Node(sm)

        # insert Edges into Nodes
        for edge in edges:
            for sm in edge.get_submols():
                nodes_dict[sm].add_edge(edge)

        # insert Nodes into Edges
        for node in nodes_dict.values():
            for edge in node.get_edges():
                edge.add_node(node)

        self._sort_connections(nodes_dict)
        return nodes_dict

    @staticmethod
    def _sort_connections(nodes_dict: dict[SubMolecule,]):
        """
        This function gets an unsorted nodes_dict and geometrically sorts them based on their size and position.
        For each Node(Submolecule) the edges(connection_point) will be sorted.
        Repeated till every node is added.
        :param nodes_dict: Dictionary mapping Submolecules to their Nodes.
        :return: Nothing is returned, the changes are made by reference.
        """
        for sm, node in nodes_dict.items():
            edges = node.get_edges()
            if len(edges) <= 1:
                # nothing to sort
                continue
            # create a dictionary mapping atom_idx/connection_point to edge & number of atoms
            edge_size_dict = {}
            for edge in edges:
                size = sum([len(n.get_submol().get_atoms()) for n in node.get_downstream_nodes(edge)])
                if edge.get_idx() not in edge_size_dict:
                    edge_size_dict[edge.get_idx()] = []
                edge_size_dict[edge.get_idx()].append((size, edge))

            new_edges = sm.sort_connections(edge_size_dict)
            if new_edges:
                node.set_edges(new_edges)

    def get_connections(self):
        """
        :return: A dictionary mapping atom indices with connected SubMolecules
        """
        connection_dict = {}
        # For each SubMol
        for sub_mol in self._mol_class.get_sub_molecules():
            # get connection points
            for atom_idx in sub_mol.get_con_points():
                # safe them in a dict
                if atom_idx not in connection_dict:
                    connection_dict[atom_idx] = []
                connection_dict[atom_idx].append(sub_mol)
        return connection_dict

    class Node:
        """
        This class implements the function for nodes in the graph
        """

        def __init__(self, submol: SubMolecule):
            self._submol = submol
            self.edges = []

        def add_edge(self, edge):
            """
            Appends a new edge
            :param edge: an Edge object
            :return: None
            """
            self.edges.append(edge)

        def get_degree(self):
            """
            :return: The degree / number of edges of this node
            """
            return len(self.get_edges())

        def get_submol(self):
            """
            :return: The SubMolecule of this node
            """
            return self._submol

        def get_edges(self):
            """
            :return: A list of all edges
            """
            return self.edges

        def set_edges(self, edges):
            """
            Overwrite the list edges
            :param edges: a new list of edges
            :return: None
            """
            self.edges = edges

        def get_downstream_nodes(self, edge, visited_nodes=None, invert_edge=True):
            """
            This function gets all nodes, which can be accessed when moving down the edge.
            Nodes can only be visited once.
            :param edge: The edge which this function moves to.
            :param visited_nodes: a list of all visited nodes. Default should always be None
            :param invert_edge: a bool with allows you to invert the edges. When calling should always be True
            :return: a list of nodes
            """
            # "cutting" at edge and then returning remaining connected Nodes recursively
            nodes = []
            edges = []
            if not visited_nodes:
                # First Iteration
                visited_nodes = [self]

            if invert_edge:
                edges.append(edge)
            else:
                edges = self.get_edges().copy()
                # "cutting" all edges with same idx as edge
                removed = 0
                for i in range(len(edges)):
                    if edges[i - removed].get_idx() == edge.get_idx():
                        edges.pop(i - removed)
                        removed += 1

            # print(f"Downstream: {self}")
            for edge in edges:
                # get every connected Node
                node = edge.get_other_node(self)
                if node not in visited_nodes:
                    visited_nodes.append(node)
                    nodes.append(node)
                    nodes = nodes + node.get_downstream_nodes(edge, visited_nodes, invert_edge=False)

            return nodes

        def __eq__(self, other):
            if type(other) is SubMolecule:
                return self.get_submol() == other
            else:
                return self.get_submol() == other.get_submol()

        def __str__(self):
            sm = self.get_submol()
            return f"SM: {sm.get_name()}\t" \
                   f"Edges: {[(edge.get_idx(), edge.get_other_sm(sm).get_atoms()) for edge in self.get_edges()]}"

        def __repr__(self):
            return "\n" + self.__str__()

    class Edge:
        """
        This class implements the function for edges in the graph
        """

        def __init__(self, idx, sub_mol_1, sub_mol_2):
            self._idx = idx
            self._sub_mol_1 = sub_mol_1
            self._sub_mol_2 = sub_mol_2
            self._node_1 = None
            self._node_2 = None

        def get_idx(self):
            """
            :return: The index of the atom/edge.
            """
            return self._idx

        def get_other_sm(self, this):
            """
            :param this: one of the Submolecule
            :return: The other Submolecule
            """
            if self._sub_mol_1 is this:
                return self._sub_mol_2
            elif self._sub_mol_2 is this:
                return self._sub_mol_1

        def get_other_node(self, this):
            """
            :param this: one of the Nodes
            :return: The other Nodes
            """
            if not self._node_1 or not self._node_2:
                raise ValueError
            if self._node_1 is this:
                return self._node_2
            elif self._node_2 is this:
                return self._node_1

        def get_nodes(self):
            """
            :return: a tuple of both Nodes
            """
            return self._node_1, self._node_2

        def get_submols(self):
            """
            :return: a tuple of both SubMolecules
            """
            return self._sub_mol_1, self._sub_mol_2

        def add_node(self, node):
            """
            Adds a Node and check if SubMolecule is the same
            :param node: a Node
            :return: None
            """
            if node.get_submol() is self._sub_mol_1:
                self._node_1 = node
            elif node.get_submol() is self._sub_mol_2:
                self._node_2 = node
            else:
                raise ValueError
