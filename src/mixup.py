import copy
import random
import numpy as np
from numpy.random import default_rng
from io import StringIO
from skbio import TreeNode
from skbio.diversity.beta import weighted_unifrac, unweighted_unifrac


class Mixup:
    """
    A class for applying Mixup and Cutmix augmentations to Phylogeny datasets using a tree-based structure.
    It handles the mixing of samples based on the Aitchison distance and tree-based relationships.

    Attributes:
        data (PhylogenyDataset): The deep-copied Phylogeny dataset to be augmented.
        taxonomy_tree (PhylogenyTree): A tree representing the hierarchical structure of features in the taxonomy.
        phylogeny_tree (PhylogenyTree): A tree representing the hierarchical structure of features in the phylogeny.
        all_nodes_taxo (list): A list of nodes in the taxonomy tree.
        all_nodes_phyl (list): A list of nodes in the phylogeny tree.
        distances (np.ndarray): Euclidean distances between samples in the dataset.
        unifrac_distances (np.ndarray): Unifrac distances between samples in the dataset.
        taxonomy_leaves (list): Leaf names of the taxonomy tree.
        phylogeny_leaves (list): Leaf names of the phylogeny tree.
    """
    
    def __init__(self, dataset, taxonomy_tree, phylogeny_tree, contrastive_learning=False):
        """
        Initialize the Mixup object.

        :param dataset: PhylogenyDataset instance for the dataset to augment.
        :param taxonomy_tree: PhylogenyTree instance representing the taxonomy feature hierarchy.
        :param phylogeny_tree: PhylogenyTree instance representing the phylogeny feature hierarchy.
        :param contrastive_learning: Boolean indicating whether contrastive learning mode is enabled.
        """
        self.data = copy.deepcopy(dataset)
        self.taxonomy_tree = taxonomy_tree
        self.phylogeny_tree = phylogeny_tree
        self.phylogeny_tree.ete_tree.resolve_polytomy(recursive=True)

        self.all_nodes_taxo = list(self.taxonomy_tree.ete_tree.traverse("levelorder"))
        self.all_nodes_phyl = list(self.phylogeny_tree.ete_tree.traverse("levelorder"))

        if contrastive_learning:
            np.random.seed(None)

        # Compute distances
        self.taxonomy_leaves = list(self.taxonomy_tree.ete_tree.leaf_names())
        self.phylogeny_leaves = list(self.phylogeny_tree.ete_tree.leaf_names())
        self.distances = self._compute_euclidean_distances()
        self.unifrac_distances = dataset.unifrac_distances_map

    def _compute_euclidean_distances(self):
        """
        Compute the Euclidean distances between all pairs of samples in the dataset.
        """
        distances = np.zeros((len(self.data), len(self.data)))
        for i in range(len(self.data)):
            for j in range(i + 1, len(self.data)):
                distances[i, j] = self._euclidean_distance(self.data.X[i], self.data.X[j])
        return distances

    def _compute_eligible_pairs(self, min_threshold, max_threshold):
        """
        Compute the eligible sample pairs for Mixup based on the Euclidean distance thresholds.

        :param min_threshold: A float representing the minimum Euclidean distance threshold.
        :param max_threshold: A float representing the maximum Euclidean distance threshold.
        :return: A list of eligible sample pairs.
        """
        eligible_pairs = []
        for i in range(len(self.data)):
            for j in range(i + 1, len(self.data)):
                if min_threshold <= self.distances[i, j] <= max_threshold:
                    eligible_pairs.append((i, j))
        return eligible_pairs

    def _select_samples(self, eligible_pairs, num_samples):
        """
        Select sample pairs for Mixup without replacement.

        :param eligible_pairs: A list of eligible sample pairs for Mixup.
        :param num_samples: The number of Mixup samples to generate.
        :return: A list of selected sample pairs.
        :raises ValueError: If no sample pairs are found within the distance threshold.
        """
        if not eligible_pairs:
            raise ValueError("No sample pairs found with distance below the threshold")

        selected_pairs = random.choices(eligible_pairs, k=num_samples)
        return selected_pairs

    def _euclidean_distance(self, x, y):
        """
        Compute the Euclidean distance between two samples.

        :param x: A numpy array representing the first sample.
        :param y: A numpy array representing the second sample.
        :return: The Euclidean distance between the two samples.
        """
        return np.linalg.norm(x - y, ord=2)

    def _aitchison_addition(self, x, v):
        """
        Perform Aitchison addition on two samples.

        :param x: A numpy array representing the first sample.
        :param v: A numpy array representing the second sample.
        :return: The result of Aitchison addition.
        """
        sum_xv = np.sum(x * v)
        return (x * v) / sum_xv

    def _aitchison_scalar_multiplication(self, lam, x):
        """
        Perform Aitchison scalar multiplication on a sample.

        :param lam: A float representing the scalar.
        :param x: A numpy array representing the sample.
        :return: The result of Aitchison scalar multiplication.
        """
        sum_xtolam = np.sum(x ** lam)
        return (x ** lam) / sum_xtolam

    def _mixup_samples(self, idx1, idx2, lam, method, tree, contrastive_learning=False):
        """
        Perform Mixup on a pair of samples.

        :param idx1: The index of the first sample.
        :param idx2: The index of the second sample.
        :param lam: A float representing the lambda parameter for Mixup.
        :param method: A string representing the Mixup method to use.
        :param tree: A string indicating the type of tree used ('taxonomy' or 'phylogeny').
        :param contrastive_learning: Boolean indicating if contrastive learning mode is enabled.
        :return: The mixed sample and its label.
        """
        if contrastive_learning:
            np.random.seed(0)

        x1, y1 = self.data[idx1]
        x2, y2 = self.data[idx2]

        if method == 'vanilla':
            mixed_x = lam * x1 + (1 - lam) * x2
        elif method == 'aitchison':
            mixed_x = self._aitchison_addition(
                self._aitchison_scalar_multiplication(lam, x1),
                self._aitchison_scalar_multiplication(1 - lam, x2)
            )
        elif method == 'phylomix':
            if tree == 'taxonomy':
                all_nodes = self.all_nodes_taxo
            else:
                all_nodes = self.all_nodes_phyl

            features_dict = {feature_name: idx for idx, feature_name in enumerate(self.data.features.tolist())}
            n_leaves = int((1 - lam) * len(self.taxonomy_leaves))
            selected_index = set()
            mixed_x = x1.copy()

            while len(selected_index) < n_leaves:
                available_node = random.choice(all_nodes)
                leaf_names = available_node.leaf_names()
                leaf_idx = [features_dict[leaf_name] for leaf_name in leaf_names]
                selected_index.update(leaf_idx)

            selected_index = random.sample(list(selected_index), n_leaves)
            leaf_counts1, leaf_counts2 = x1[selected_index].astype(np.float64), x2[selected_index].astype(np.float64)
            total1, total2 = leaf_counts1.sum(), leaf_counts2.sum()
            if total2 > 0 and total1 > 0:
                leaf_counts2_normalized = leaf_counts2 / total2
                new_counts = (total1 * leaf_counts2_normalized).astype(int)
                mixed_x[selected_index] = new_counts
            else:
                mixed_x[selected_index] = leaf_counts1

        mixed_y = lam * y1 + (1 - lam) * y2
        return mixed_x, mixed_y

    def generate_points(self, N):
        """
        Generate random points from a Dirichlet distribution.

        :param N: The number of points to generate.
        :return: A numpy array of generated points.
        """
        alpha = [1 for _ in range(N)]
        return np.random.dirichlet(alpha)

    def general_mixup(self, min_threshold, max_threshold, num_samples):
        """
        Apply a generak Mixup using all features.

        :param min_threshold: The minimum threshold for sample selection.
        :param max_threshold: The maximum threshold for sample selection.
        :param num_samples: The number of Mixup samples to generate.
        :return: The augmented dataset.
        """
        n_rows = len(self.data.y)
        combination_matrix = np.zeros((num_samples, n_rows))
        for i in range(num_samples):
            selected_numbers = np.random.choice(n_rows, size=4, replace=False)
            combination_matrix[i][selected_numbers] = self.generate_points(4)

        augmented_data_x = np.dot(combination_matrix, self.data.X).astype(np.float32)
        augmented_data_y = np.dot(combination_matrix, self.data.y).astype(np.float32)

        self.data.X = np.vstack((self.data.X, augmented_data_x))
        self.data.y = np.vstack((self.data.y, augmented_data_y))

        return self.data

    def intra_mixup(self, min_threshold, max_threshold, method, num_samples, alpha):
        """
        Perform intra-class Mixup on the dataset.

        :param min_threshold: The minimum threshold for sample selection.
        :param max_threshold: The maximum threshold for sample selection.
        :param method: The Mixup method to use ('intra_vanilla' or 'intra_aitchison').
        :param num_samples: The number of Mixup samples to generate.
        :param alpha: The alpha parameter for the Beta distribution.
        :return: The augmented dataset.
        """
        classes = np.unique(self.data.y)
        class_data = {cls: [] for cls in classes}

        for cls in classes:
            index = np.where(self.data.y == cls)[0]
            class_data[cls] = self.data.X[index]

        mixed_xs, mixed_ys = [], []
        while len(mixed_xs) < num_samples:
            class_drawn = np.random.choice(classes)
            data = class_data[class_drawn]
            indices_drawn = np.random.choice(len(data), size=2, replace=False)
            idx1, idx2 = indices_drawn[0], indices_drawn[1]
            lam = np.random.beta(alpha, alpha)
            x1, x2 = data[idx1], data[idx2]
            if method == 'intra_vanilla':
                mixed_x = lam * x1 + (1 - lam) * x2
            elif method == 'intra_aitchison':
                mixed_x = self._aitchison_addition(
                    self._aitchison_scalar_multiplication(lam, x1),
                    self._aitchison_scalar_multiplication(1 - lam, x2)
                )
            mixed_ys.append(class_drawn)
            mixed_xs.append(mixed_x)

        self.data.X = np.vstack((self.data.X, np.array(mixed_xs)))
        self.data.y = np.concatenate((self.data.y, np.array(mixed_ys)))

        return self.data

    def mixup(self, min_threshold, max_threshold, num_samples, method, alpha, tree,
              index1=None, index2=None, contrastive_learning=False, seed=0):
        """
        Perform Mixup augmentation on the dataset.

        :param min_threshold: A float representing the minimum Euclidean distance for selecting sample pairs.
        :param max_threshold: A float representing the maximum Euclidean distance for selecting sample pairs.
        :param num_samples: The number of Mixup samples to generate.
        :param method: A string representing the Mixup method to use.
        :param alpha: The alpha parameter for the Beta distribution.
        :param tree: The type of tree used ('taxonomy' or 'phylogeny').
        :param index1: Optional list of specific indices for the first sample.
        :param index2: Optional list of specific indices for the second sample.
        :param contrastive_learning: Boolean indicating if contrastive learning mode is enabled.
        :param seed: Random seed for reproducibility in contrastive learning mode.
        :return: An augmented PhylogenyDataset instance.
        :raises ValueError: If no sample pairs are found within the distance threshold.
        """
        eligible_pairs = self._compute_eligible_pairs(min_threshold, max_threshold)
        mixed_xs, mixed_ys = [], []

        if index1 is not None and index2 is not None:
            selected_pairs = list(zip(index1, index2))
        else:
            selected_pairs = self._select_samples(eligible_pairs, num_samples)

        for idx1, idx2 in selected_pairs:
            if contrastive_learning:
                rng = default_rng(seed=seed)
                lam = rng.beta(2, 2)
            else:
                lam = np.random.beta(alpha, alpha)
            mixed_x, mixed_y = self._mixup_samples(idx1, idx2, lam, method, tree, contrastive_learning)
            mixed_xs.append(mixed_x)
            mixed_ys.append(mixed_y)

        if contrastive_learning:
            return np.array(mixed_xs)
        else:
            self.data.X = np.vstack((self.data.X, np.array(mixed_xs)))
            self.data.y = np.vstack((self.data.y, np.array(mixed_ys)))
            return self.data

    def _cutmix_with_subtree(self, idx1, idx2, subtree_nodes):
        """
        Perform Cutmix on a pair of samples by swapping the data of leaves in the specified subtrees.

        :param idx1: The index of the first sample.
        :param idx2: The index of the second sample.
        :param subtree_nodes: A list of nodes in the tree representing the subtrees to swap.
        :return: The swapped samples.
        """
        x1, y1 = self.data[idx1]
        x2, y2 = self.data[idx2]

        num_swapped_leaves = 0
        for node in subtree_nodes:
            leaf_names = node.leaf_names()
            leaf_idx = [self.data.features.tolist().index(leaf_name) for leaf_name in leaf_names]
            num_swapped_leaves += len(leaf_idx)
            x1_orig, x2_orig = x1[leaf_idx].copy(), x2[leaf_idx].copy()
            x1[leaf_idx], x2[leaf_idx] = x2_orig, x1_orig
            x1[leaf_idx] = x1[leaf_idx] * x1_orig.sum() / x1[leaf_idx].sum()
            x2[leaf_idx] = x2[leaf_idx] * x2_orig.sum() / x2[leaf_idx].sum()

        y1 = y1 * (1 - num_swapped_leaves / len(self.data.features)) + y2 * (num_swapped_leaves / len(self.data.features))
        y2 = y2 * (1 - num_swapped_leaves / len(self.data.features)) + y1 * (num_swapped_leaves / len(self.data.features))

        return x1, y1, x2, y2

    def _compute_available_nodes(self, height):
        """
        Compute available nodes at a given height in the tree.

        :param height: The height in the tree.
        :return: A list of available nodes at the specified height.
        """
        return [ete_node for ete_node in self.taxonomy_tree.ete_tree.traverse("levelorder")
                if self.taxonomy_tree.depths[ete_node.name] == height]

    def compositional_cutmix(self, min_threshold, max_threshold, num_samples):
        """
        Perform compositional cutmix augmentation on the dataset.

        :param min_threshold: The minimum threshold for sample selection.
        :param max_threshold: The maximum threshold for sample selection.
        :param num_samples: The number of samples to generate.
        :return: The augmented dataset.
        """
        mixed_xs, mixed_ys = [], []
        classes = [0, 1]
        class_data = {cls: [] for cls in classes}

        for cls in classes:
            index = np.where(self.data.y == cls)[0]
            class_data[cls] = self.data.X[index]

        while len(mixed_xs) < num_samples:
            class_drawn = np.random.choice(classes)
            data = class_data[class_drawn]
            lam = np.random.uniform(0, 1)
            indices_drawn = np.random.choice(len(data), size=2, replace=False)
            idx1, idx2 = indices_drawn[0], indices_drawn[1]
            x1, x2 = data[idx1], data[idx2]
            mixed_x = x1.copy()
            for j in range(len(self.data.features)):
                Ij = np.random.binomial(n=1, p=lam)
                mixed_x[j] = x1[j] if Ij == 1 else x2[j]
            mixed_xs.append(mixed_x)
            mixed_ys.append(class_drawn)

        self.data.X = np.vstack((self.data.X, np.array(mixed_xs)))
        self.data.y = np.concatenate((self.data.y, np.array(mixed_ys)))

        return self.data

    def intra_cutmix(self, min_threshold, max_threshold, num_samples, height, num_subtrees):
        """
        Perform Cutmix augmentation within the same class by swapping subtrees.

        :param min_threshold: The minimum threshold for sample selection.
        :param max_threshold: The maximum threshold for sample selection.
        :param num_samples: The number of samples to generate.
        :param height: The height in the tree at which to swap subtrees.
        :param num_subtrees: The number of subtrees to swap between samples.
        :return: The augmented dataset.
        """
        eligible_pairs = self._compute_eligible_pairs(min_threshold, max_threshold)
        mixed_xs, mixed_ys = [], []
        available_nodes = self._compute_available_nodes(height)
        classes = [0, 1]
        class_data = {cls: [] for cls in classes}

        for cls in classes:
            index = np.where(self.data.y == cls)[0]
            class_data[cls] = self.data.X[index]

        features_dict = {feature_name: idx for idx, feature_name in enumerate(self.data.features.tolist())}
        leaf_names_list = [node.leaf_names() for node in available_nodes]
        leaf_idx_list = [[features_dict[leaf_name] for leaf_name in leaf_names] for leaf_names in leaf_names_list]

        while len(mixed_xs) < num_samples:
            class_drawn = np.random.choice(classes)
            data = class_data[class_drawn]
            indices_drawn = np.random.choice(len(data), size=2, replace=False)
            idx1, idx2 = indices_drawn[0], indices_drawn[1]
            x1, x2 = data[idx1], data[idx2]
            swap_nodes, leaf_idxes = [], []

            while len(swap_nodes) < num_subtrees:
                available_node = random.choice(available_nodes)
                leaf_idx = leaf_idx_list[available_nodes.index(available_node)]
                if (x1[leaf_idx] != 0).any() or (x2[leaf_idx] != 0).any():
                    swap_nodes.append(available_node)
                    leaf_idxes.extend(leaf_idx)

            mixed_x = x1.copy()
            mixed_x[leaf_idxes] = x2[leaf_idxes]
            mixed_xs.append(mixed_x)
            mixed_ys.append(class_drawn)

        self.data.X = np.vstack((self.data.X, np.array(mixed_xs)))
        self.data.y = np.concatenate((self.data.y, mixed_ys))

        return self.data
