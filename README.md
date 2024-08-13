# phylomix

## Dataset

To access dataset: [Google Drive](https://drive.google.com/drive/folders/1fAzZKaI7mMx0xZGI7dOwEfCa1gRNkmCm).

Our experiement contains TADA augmentation, please refer to the repo [TADA](https://github.com/tada-alg/TADA). And add the augmented data file into the data folders.

## Requirements

Required packages are listed in ```environment.yml```. You can create a conda environment with the following commant:

```{bash}
conda env create -f environment.yml
```

## Usage

- Specify arguments in an argument file
- To run supervised learning training:

```{bash}
bash run_job.sh args_file.txt supervised
```

- To run contrastive learning training:

```{bash}
bash run_job.sh args_file.txt contrastive
```




