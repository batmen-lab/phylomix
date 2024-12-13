# README: Mixup and Phylomix for Phylogeny Datasets

## Overview
This project provides an implementation of data augmentation methods for phylogeny datasets, with a focus on leveraging tree-based relationships between features. The main method introduced is **Phylomix**, which combines phylogenetic information with compositional data for enhanced data augmentation. In addition, several baseline methods are implemented, including vanilla Mixup, Aitchison Mixup, and Cutmix variants.

The augmentation methods aim to improve model generalization by creating synthetic samples based on relationships between existing samples. These methods are particularly suitable for hierarchical and compositional datasets.

---

## Features
1. **Phylomix**: A novel method using phylogeny or taxonomy trees for Mixup.
2. **Vanilla Mixup**: Traditional Mixup with linear interpolation.
3. **Aitchison Mixup**: Mixup based on Aitchison geometry for compositional data.
4. **Compositional Cutmix**: Variants of Cutmix adapted to phylogenetic trees.
5. **Aitchison Mixup**: Mixup in the Aitchison space.

---

## Dataset

To access dataset: [Google Drive](https://drive.google.com/drive/folders/1fAzZKaI7mMx0xZGI7dOwEfCa1gRNkmCm).

Our experiment contains TADA augmentation, please refer to the repo [TADA](https://github.com/tada-alg/TADA). Add the augmented data file into the data folders.

For MB-GAN, please refer to the repo [MB-GAN](https://github.com/zhanxw/MB-GAN). And we provide a script to prepare our dataset in the format of MB-GAN. Please run:

```bash
python src/gan_data_prepare.py
```

And it will prepare the data in the format of required by MB-GAN.

---

## Requirements

Required packages are listed in `environment.yml`. You can create a conda environment with the following command:

```bash
conda env create -f environment.yml
```

---

## Classes and Methods

### `Mixup`
The `Mixup` class encapsulates all Mixup and Cutmix augmentation methods.

### Constructor
```python
Mixup(dataset, taxonomy_tree, phylogeny_tree, contrastive_learning=False)
```
**Parameters:**
- `dataset`: A `PhylogenyDataset` instance containing data to augment.
- `taxonomy_tree`: A `PhylogenyTree` instance for taxonomy.
- `phylogeny_tree`: A `PhylogenyTree` instance for phylogeny.
- `contrastive_learning`: (Optional) Enable contrastive learning mode (default: `False`).

### Methods

#### `mixup`
Performs Mixup augmentation.
```python
mixup(num_samples, method, alpha, tree, min_threshold=None, max_threshold=None,
      index1=None, index2=None, contrastive_learning=False, seed=0)
```
**Parameters:**
- `num_samples`: Number of Mixup samples to generate.
- `method`: Mixup method (`vanilla`, `aitchison`, `phylomix`).
- `alpha`: Beta distribution parameter for sample mixing.
- `tree`: Type of tree to use (`taxonomy` or `phylogeny`).
- Additional parameters for specific indices and thresholds.

**Returns:** Augmented `PhylogenyDataset`.

#### `compositional_cutmix`
Performs Cutmix augmentation based on compositional data.
```python
compositional_cutmix(num_samples, min_threshold=None, max_threshold=None)
```

#### `intra_mixup`
Performs Mixup within the same class.
```python
intra_mixup(min_threshold, max_threshold, method, num_samples, alpha)
```

#### `intra_cutmix`
Performs intra-class Cutmix by swapping subtrees.
```python
intra_cutmix(min_threshold, max_threshold, num_samples, height, num_subtrees)
```

---

## Training

- Specify arguments in an argument file.
- To run supervised learning training:

```bash
bash run_job.sh args_file.txt supervised
```

- To run contrastive learning training:

```bash
bash run_job.sh args_file.txt contrastive
```

---

## Usage

### 1. Setting Up the Data

The `setup_data` function initializes the dataset and the phylogeny tree from the specified file paths. This function is designed to make it easy to load and prepare your data for augmentation, and we prune the tree leaves to match the number of data features.

```python
from mixup import setup_data

# File paths to your data and phylogeny tree
data_fp = 'path/to/your/data_file.tsv.xz'
meta_fp = 'path/to/your/meta_file.tsv'
target_fp = 'path/to/your/target_file.py'
phylogeny_tree_fp = 'path/to/your/phylogeny_tree.nwk'

# Initialize the dataset and phylogeny tree
data, tree = setup_data(data_fp, meta_fp, target_fp, phylogeny_tree_fp, prune=True)
```

### 2. Augmenting the Data
Once you have your dataset and tree set up, you can use the `augment` function to apply a variety of augmentation techniques. This function allows you to easily apply mixup-based augmentations.

```python
from mixup import augment

augmented_data = augment(
    data=data,
    phylogeny_tree=tree,
    num_samples=3.0,   # Augment to 3 times the original number of samples
    alpha=2.0,
    aug_type='phylomix'
)
```

### 3. Applying Baseline Methods

#### Vanilla Mixup
Vanilla Mixup uses linear interpolation between two samples.
```python
augmented_dataset = mixup_instance.mixup(
    num_samples=100,
    method='vanilla',
    alpha=0.5,
    tree='taxonomy'
)
```

#### Aitchison Mixup
Aitchison Mixup applies Mixup using the Aitchison geometry, which is specifically suited for compositional data.
```python
augmented_dataset = mixup_instance.mixup(
    num_samples=100,
    method='aitchison',
    alpha=0.5,
    tree='taxonomy'
)
```

#### Compositional Cutmix
Compositional Cutmix swaps data between samples in a way that respects compositional constraints.
```python
augmented_dataset = mixup_instance.compositional_cutmix(num_samples=50)
```

#### Intra-Class Mixup
This baseline performs Mixup within the same class, ensuring that generated samples are class-consistent.
```python
augmented_data = mixup_instance.intra_mixup(
    min_threshold=0.1,
    max_threshold=0.8,
    method='intra_aitchison',
    num_samples=150,
    alpha=0.6
)
```

#### Intra-Class Cutmix
Intra-Class Cutmix swaps specific subtrees within the same class, respecting hierarchical structures.
```python
augmented_dataset = mixup_instance.intra_cutmix(
    min_threshold=0.1,
    max_threshold=0.5,
    num_samples=50,
    height=3,
    num_subtrees=2
)
```

---

## Examples

### Example 1: Phylomix Augmentation
```python
augmented_data = mixup_instance.mixup(
    num_samples=200,
    method='phylomix',
    alpha=0.3,
    tree='phylogeny'
)
```

### Example 2: Intra-Class Aitchison Mixup
```python
augmented_data = mixup_instance.intra_mixup(
    min_threshold=0.1,
    max_threshold=0.8,
    method='intra_aitchison',
    num_samples=150,
    alpha=0.6
)
```
---


## Notes
- Ensure that the taxonomy and phylogeny trees have the same leaves; otherwise, prune them.
- Use contrastive learning mode for unsupervised tasks.
- Experiment with `alpha` to control the degree of interpolation.

---

## License
This project is open-source and licensed under the MIT License.

