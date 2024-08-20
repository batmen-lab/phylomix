# phylomix

## Dataset

To access dataset: [Google Drive](https://drive.google.com/drive/folders/1fAzZKaI7mMx0xZGI7dOwEfCa1gRNkmCm).

Our experiement contains TADA augmentation, please refer to the repo [TADA](https://github.com/tada-alg/TADA). And add the augmented data file into the data folders.

## Requirements

Required packages are listed in ```environment.yml```. You can create a conda environment with the following commant:

```{bash}
conda env create -f environment.yml
```

## Training

- Specify arguments in an argument file
- To run supervised learning training:

```{bash}
bash run_job.sh args_file.txt supervised
```

- To run contrastive learning training:

```{bash}
bash run_job.sh args_file.txt contrastive
```

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






