import numpy as np
import scipy.sparse as sp

from collections import Counter

from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer, normalize


def to_binary_bag_of_words(features):
    features_copy = features.tocsr()
    features_copy.data[:] = 1.0
    return features_copy


def normalize_adj(A):
    # Make sure that there are no self-loops
    A = eliminate_self_loops(A)
    D = np.ravel(A.sum(1))
    D[D == 0] = 1  # avoid division by 0 error
    D_sqrt = np.sqrt(D)
    return A / D_sqrt[:, None] / D_sqrt[None, :]


def renormalize_adj(A):
    A_tilde = A.tolil()
    A_tilde.setdiag(1)
    A_tilde = A_tilde.tocsr()
    A_tilde.eliminate_zeros()
    D = np.ravel(A.sum(1))
    D_sqrt = np.sqrt(D)
    return A / D_sqrt[:, None] / D_sqrt[None, :]


def row_normalize(matrix):
    return normalize(matrix, norm='l1', axis=1)


def add_self_loops(A, value=1.0):
    A = A.tolil()  # make sure we work on a copy of the original matrix
    A.setdiag(value)
    A = A.tocsr()
    if value == 0:
        A.eliminate_zeros()
    return A


def eliminate_self_loops(A):
    A = A.tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


def largest_connected_components(sparse_graph, n_components=1):
    """ selects the largest connected component in the graph.

    Parameter
    ----------
    Sparse_graph: sparse graph
        Enter the map.
    N_components:int, default value is 1
        The largest connected components to keep.

    Return
    -------
    Sparse_graph: sparse graph
        Enter a submap of the graph where only the nodes in the largest n_components are retained"""
    
    _, component_indices = sp.csgraph.connected_components(sparse_graph.adj_matrix)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    return create_subgraph(sparse_graph, nodes_to_keep=nodes_to_keep)


def create_subgraph(sparse_graph, _sentinel=None, nodes_to_remove=None, nodes_to_keep=None):
    """
    Create a graph with the specified subset of nodes.

    One of (nodes_to_remove, nodes_to_keep) should be provided, while the other remains None.
    Note that to avoid confusion, you need to pass the node index as a named parameter to this function.

    Parameter
    ----------
    Sparse_graph:SparseGraph
        Enter the map.
    _sentinel: none
        Internal to prevent the transfer of positional parameters. Do not use.
    Nodes_to_remove: an array similar to int
        The node index that must be deleted.
    Nodes_to_keep: an array similar to int
        The index of the node that must be kept.

    Return
    -------
    Sparse_graph:SparseGraph
        The chart for the specified node has been deleted.
    """
    if _sentinel is not None:
        raise ValueError("Only call `create_subgraph` with named arguments',"
                         " (nodes_to_remove=...) or (nodes_to_keep=...)")
    if nodes_to_remove is None and nodes_to_keep is None:
        raise ValueError("Either nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None and nodes_to_keep is not None:
        raise ValueError("Only one of nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None:
        nodes_to_keep = [i for i in range(sparse_graph.num_nodes()) if i not in nodes_to_remove]
    elif nodes_to_keep is not None:
        nodes_to_keep = sorted(nodes_to_keep)
    else:
        raise RuntimeError("This should never happen.")

    sparse_graph.adj_matrix = sparse_graph.adj_matrix[nodes_to_keep][:, nodes_to_keep]
    if sparse_graph.attr_matrix is not None:
        sparse_graph.attr_matrix = sparse_graph.attr_matrix[nodes_to_keep]
    if sparse_graph.labels is not None:
        sparse_graph.labels = sparse_graph.labels[nodes_to_keep]
    if sparse_graph.node_names is not None:
        sparse_graph.node_names = sparse_graph.node_names[nodes_to_keep]
    return sparse_graph


def binarize_labels(labels, sparse_output=False, return_classes=False):
    """converts the label vector to a binary label matrix.

    In the default single-label case, the label looks like
    Labels = [y1,y2,y3,...].
    Multi-label formats are also supported.
    In this case, the label should look like this
    Labels = [[y11,y12],[y21,y22,y23],[y31],...].

    Parameter
    ----------
    Label: similar to array, shape [num_samples]
        An array of node labels in a single-label or multi-label format.
    Sparse_output: bool, default is False
        Whether to return label_matrix in CSR format.
    Return_classes:bool, defaults to False
        Whether to return the class corresponding to the column of the label matrix.

    Return
    -------
    Label_matrix:np.ndarray or sp.csr_matrix,shape [num_samples,num_classes]
        The binary matrix of the class label.
    Num_classes = the number of unique values ​​in the "labels" array.
        Label_matrix [i,k] = 1 <=>Node i belongs to class k.
    Classes:np.array,shape [num_classes], optional
        The class corresponding to each column of label_matrix.

    """
    if hasattr(labels[0], '__iter__'):  # labels[0] is iterable <=> multilabel format
        binarizer = MultiLabelBinarizer(sparse_output=sparse_output)
    else:
        binarizer = LabelBinarizer(sparse_output=sparse_output)
    label_matrix = binarizer.fit_transform(labels).astype(np.float32)
    return (label_matrix, binarizer.classes_) if return_classes else label_matrix


def remove_underrepresented_classes(g, train_examples_per_class, val_examples_per_class):
    """Remove the node corresponding to the less than class from the graph
    Num_classes * train_examples_per_class + num_classes * val_examples_per_class nodes.

    Otherwise these cases will disrupt the training process.
    """
    min_examples_per_class = train_examples_per_class + val_examples_per_class
    examples_counter = Counter(g.labels)
    keep_classes = set(class_ for class_, count in examples_counter.items() if count > min_examples_per_class)
    keep_indices = [i for i in range(len(g.labels)) if g.labels[i] in keep_classes]

    return create_subgraph(g, nodes_to_keep=keep_indices)
