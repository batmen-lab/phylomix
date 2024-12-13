from data import PhylogenyDataset, PhylogenyTree
import numpy as np
import pickle

def adjmatrix_to_dense(x, shape, val=1):
    mask = np.zeros(shape)
    x = np.array(x).transpose()
    mask[tuple(x)] = val
    return mask

datasets = ["asd", "ibd200", "rumc", "alzbiom", "hmp2", "gd"]
targets = ["stage", "type", "pd", "ad", "type", "cohort"]
for data, target in zip(datasets, targets):
    data_fp = f'../data/{data}/data.tsv.xz'
    meta_fp = f'../data/{data}/meta.tsv'
    target_fp = f'../data/{data}/{target}.py'
    dataset = PhylogenyDataset.init_from_files(data_fp, meta_fp, target_fp)
    tree_fp2 = '../data/WoL2/phylogeny.nwk'
    tree2 = PhylogenyTree.init_from_nwk(tree_fp2)
    tree2.prune(dataset.features)

    res = {}
    X = dataset.X
    y = dataset.y
    features = dataset.features
    adj, taxa_indices = dataset.expand_phylo(tree2)
    res['X'] = X
    res['y'] = y
    res['taxa_list'] = features
    res['tf_matrix'] = adjmatrix_to_dense(adj, shape=(len(features), len(taxa_indices)))
    with open(f"gandata/{data}.pkl", "wb") as output_file:
        pickle.dump(res, output_file)


    

