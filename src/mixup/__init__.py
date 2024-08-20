from .mixup import Mixup
from .data import PhylogenyDataset, PhylogenyTree

__all__ = ['Mixup', 'PhylogenyDataset', 'PhylogenyTree', 'augment', 'setup_data']

def setup_data(data_fp, meta_fp, target_fp, phylogeny_tree_fp, prune=False):
    """
    Initializes the dataset and phylogeny tree from the provided file paths.

    Args:
        data_fp (str): File path to the dataset file.
        meta_fp (str): File path to the metadata file.
        target_fp (str): File path to the target file.
        phylogeny_tree_fp (str): File path to the phylogeny tree file.

    Returns:
        tuple: A tuple containing the initialized dataset and the pruned phylogeny tree.
    """
    # Initialize the dataset from the provided file paths
    data = PhylogenyDataset.init_from_files(data_fp, meta_fp, target_fp)
    
    # Initialize and prune the phylogeny tree
    tree = PhylogenyTree.init_from_nwk(phylogeny_tree_fp)
    if prune:
        tree.prune(data.features)
    
    return data, tree

def augment(data, phylogeny_tree=None, num_samples=1.0, alpha=2.0, aug_type='phylomix',
            min_threshold=None, max_threshold=None, clr=True, one_hot_encoding=True, normalize=False):
    """
    Augments the dataset using the specified augmentation method.

    Args:
        data (PhylogenyDataset): The dataset to be augmented.
        phylogeny_tree (PhylogenyTree, optional): The phylogeny tree used for augmentation. Defaults to None.
        num_samples (float): The ratio of the number of samples to augment. Defaults to 1.0.
        alpha (float): The alpha parameter for mixup. Defaults to 2.0.
        aug_type (str): The type of augmentation to perform. Defaults to 'phylomix'.
        min_threshold (float, optional): Minimum threshold for certain augmentations. Defaults to None.
        max_threshold (float, optional): Maximum threshold for certain augmentations. Defaults to None.
        clr (bool): Whether to apply CLR transformation. Defaults to True.
        one_hot_encoding (bool): Whether to apply one-hot encoding. Defaults to True.
        normalize (bool): Whether to normalize the dataset. Defaults to False.

    Returns:
        PhylogenyDataset: The augmented dataset.
    """
    if one_hot_encoding:
        data.one_hot_encode()
    if normalize:
        data.normalize()
    if clr:
        data.clr_transform()
    
    # Create the mixup processor
    mixup_processor = Mixup(dataset=data, taxonomy_tree=None, phylogeny_tree=phylogeny_tree)
    
    # Determine the number of samples for augmentation
    num_samples = int(data.X.shape[0] * num_samples)

    # Perform the appropriate augmentation
    if aug_type in ['vanilla', 'aitchison', 'phylomix']:
        augmented_dataset = mixup_processor.mixup(
            num_samples, 
            method=aug_type, 
            alpha=alpha, 
            tree=phylogeny_tree
        )
    elif aug_type == 'general_mixup':
        augmented_dataset = mixup_processor.general_mixup(num_samples)
    elif aug_type == 'compositional_cutmix':
        augmented_dataset = mixup_processor.compositional_cutmix(num_samples)
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")

    return augmented_dataset
