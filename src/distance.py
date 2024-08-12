import importlib
import os
import numpy as np
import pandas as pd
from ete4 import Tree
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import pickle
from skbio.diversity.beta import weighted_unifrac, unweighted_unifrac
from io import StringIO
from skbio import TreeNode

def compute_unifrac_distance_map(data_fp, meta_fp, target_fp, nwk_fp, map_fp, weighted=True):
    """
    Compute and save a Unifrac distance map for a dataset using a phylogenetic tree.

    :param data_fp: File path to the input data (tab-separated values).
    :param meta_fp: File path to the metadata (tab-separated values).
    :param target_fp: File path to the target function script.
    :param nwk_fp: File path to the Newick-formatted phylogenetic tree.
    :param map_fp: File path to save the Unifrac distance map.
    :param weighted: Boolean indicating whether to use weighted Unifrac (default is True).
    :raises ValueError: If the target function file does not exist.
    """
    # Load data and metadata
    data = pd.read_table(data_fp, index_col=0)
    meta = pd.read_table(meta_fp, index_col=0)

    # Load the target function
    if not os.path.exists(target_fp):
        raise ValueError('Target function does not exist.')
    
    spec = importlib.util.spec_from_file_location('target', target_fp)
    target_lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(target_lib)
    target = target_lib.target

    # Get target values and filter samples
    prop, name = target(meta)
    ids = [x for x in data.columns if x in set(prop.index)]
    data, prop = data[ids], prop.loc[ids]

    # Input data
    X = data.values.T.astype(np.float32)
    y = prop.values
    features = data.index.values

    # Load the phylogenetic tree
    with open(nwk_fp, 'r') as nwk_file:
        tree = Tree(nwk_file, parser=1)
    tree.name = 'root'

    # Convert the tree to skbio format
    sk_tree = TreeNode.read(StringIO(tree.write()))

    # Initialize the Unifrac distance map
    unifrac_map = np.zeros((len(X), len(X)))

    # Compute Unifrac distances
    for i in range(len(X)):
        for j in range(i, len(X)):  # Use symmetry to avoid redundant calculations
            if weighted:
                distance = weighted_unifrac(X[i], X[j], features, sk_tree)
            else:
                distance = unweighted_unifrac(X[i], X[j], features, sk_tree)
            
            unifrac_map[i, j] = distance
            unifrac_map[j, i] = distance  # Symmetry

    # Save the Unifrac distance map to a file
    with open(map_fp, 'wb') as file:
        pickle.dump(unifrac_map, file)

    print(f"Unifrac distance map saved to {map_fp}")

# Usage example
dataset = 'ibd200'
target = 'type'
data_fp = f'../data/{dataset}/data.tsv.xz'
meta_fp = f'../data/{dataset}/meta.tsv'
target_fp = f'../data/{dataset}/{target}.py'
nwk_fp = '../data/WoL2/phylogeny.nwk'
map_fp = f'../data/{dataset}/unifrac_map.pkl'

compute_unifrac_distance_map(data_fp, meta_fp, target_fp, nwk_fp, map_fp)
