import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm
from model import EncoderProjectionHead, Autoencoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import biom
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from numpy.random import default_rng
from torchmetrics.classification import (MulticlassAUROC, MulticlassAveragePrecision)
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
# import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

from supervised_learning import TrainingPipeline
from mixup.mixup import Mixup
import numpy as np
import argparse
import json
import os
import pickle
import time
from datetime import datetime
import copy


def _clr_transform(X):
    X = np.log1p(X)
    # Subtract the mean of each sample
    X = X - X.mean(axis=1, keepdims=True)
    return X

def info_nce_loss(features, batch_size, device):

    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0).to(device)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long)

    logits = logits / 0.05
    return logits.to(device), labels.to(device)

class ContrastiveLearningPipeline(TrainingPipeline):
    def __init__(self, seed, dataset):
        super().__init__(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.filepaths = {
            'TADA1': f'../../TADA/{dataset}_augmented1/augmented_data.biom',
            'TADA2': f'../../TADA/{dataset}_augmented2/augmented_data.biom',
        }
        table1, table2 = biom.load_table(self.filepaths['TADA1']), biom.load_table(self.filepaths['TADA2'])
        mat1, mat2 = table1.matrix_data.todense(), table2.matrix_data.todense()
        self.TADA_aug1, self.TADA_aug2 = np.array(mat1.T, dtype=np.float32), np.array(mat2.T, dtype=np.float32)

    def _train(self):
        # Check if data and tree are loaded
        if not self.data or not self.tree:
            raise ValueError('Please load data and tree first.')
        # Define metrics
        num_classes = len(np.unique(self.data.y))
        metrics = MetricCollection({
            'AUROC': MulticlassAUROC(num_classes=num_classes),
            'AUPRC': MulticlassAveragePrecision(num_classes=num_classes)
        })
        self.train_hparams['k_folds'] = 5
        self.train_hparams['batch_size'] = 512
        self.train_hparams['max_epochs'] = 100
        self.train_hparams['class_weight'] = 'balanced'
        
        # Define cross-validation strategy
        kf = KFold(n_splits=self.train_hparams['k_folds'], shuffle=True, random_state=self.seed)
        # Training loop
        fold_test_labels = []
        fold_test_logits = []
        for fold, (train_index, test_index) in enumerate(kf.split(self.data.X, self.data.y)):
            X_train, X_test = self.data.X[train_index], self.data.X[test_index]

            input_dim = X_train.shape[-1]
            self._initialize_unsupervised_model(input_dim, self.epochs)
            self._train_unsupervised_model(X_train)


            feature_extractor = self.unsupervised_model.encoder
            feature_extractor.eval()
            # Convert self.data.X to a PyTorch tensor and move it to the same device as the model
            X_tensor = torch.tensor(self.data.X, dtype=torch.float32).to(self.device)
            # Apply feature_extractor, ensure no gradient computation
            with torch.no_grad():
                extracted_features = feature_extractor(X_tensor)
            # Configure default training parameters
            self.aug_type = 'None'
            # Convert the extracted features back to a NumPy array
            self.data.X = extracted_features.cpu().numpy()
            # Prepare datasets
            normalize = False
            clr = False
            train_dataset = self._create_subset(self.data, 
                                                train_index, 
                                                normalize=normalize,
                                                one_hot_encoding=False, 
                                                clr=clr)
            test_dataset = self._create_subset(self.data, 
                                               test_index, 
                                               normalize=normalize,
                                               one_hot_encoding=False, 
                                               clr=clr)

            # normalize the deep features Create classifier 
            classifier = self._create_classifier(train_dataset, metrics)

                        # Set filename
            filename = f"{self.seed}_{fold}_{self.unsupervised_type}_{self.model_type}"
            filename += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            # Run training
            result = self._run_training(classifier, train_dataset, test_dataset, test_dataset, filename)
            fold_test_labels.append(torch.tensor(result['test_labels']))
            fold_test_logits.append(torch.tensor(result['test_logits']))
                     
            # Save results and model
            self._save_result(result, self.pred_dir, filename)
            # self._save_model(classifier, self.model_dir, filename)

        # Calculate metrics
        test_labels = torch.cat(fold_test_labels, dim=0)
        test_logits = torch.cat(fold_test_logits, dim=0)
        metrics.to(test_labels.device)
        test_scores = metrics(test_logits, test_labels)
        print(f"Test scores:")
        for key, value in test_scores.items():
            print(f"{key}: {value.item()}")


    def _load_simclr_data(self, train_dataset):
        mixup_processor = Mixup(train_dataset, self.tree, self.phylogeny_tree, contrastive_learning=True)
        min_threshold = np.quantile(mixup_processor.distances[mixup_processor.distances != 0], 0)
        max_threshold = np.quantile(mixup_processor.distances[mixup_processor.distances != 0], 1)
        n = train_dataset.X.shape[0]
        method = self.unsupervised_type
        idx1 = np.arange(n)
        close_idx = np.zeros(n)
        map = train_dataset.unifrac_distances_map
        mask = ~np.eye(map.shape[0], dtype=bool)
        map = map[mask].reshape(map.shape[0], -1)
        print(map.shape)
        for i in range(n):
            row = map[i]
            sorted_indices = np.argsort(row)
            midpoint = len(sorted_indices) // 2

            first_half_sorted_indices = sorted_indices[:midpoint]
            sampled_indices = np.random.choice(first_half_sorted_indices)
            close_idx[i] = sampled_indices

        close_idx = close_idx.astype(int)
        idx2 = idx1[::-1]
        if method in ['vanilla', 'phylomix', 'compositional_cutmix']:
            out1 = mixup_processor.mixup(min_threshold, max_threshold, len(idx1), method=method, alpha=2, tree='phylogeny', index1=idx1, index2=idx2, contrastive_learning=True, seed=0)
            out2 = mixup_processor.mixup(min_threshold, max_threshold, len(idx1), method=method, alpha=2, tree='phylogeny', index1=idx1, index2=idx2, contrastive_learning=True, seed=1)
        elif method == 'TADA':
            out1 = self.TADA_aug1
            out2 = self.TADA_aug2
            out1 = _clr_transform(out1)
            out2 = _clr_transform(out2)
        elif method == 'unifrac':
            out1 = train_dataset.X[idx1]
            out2 = train_dataset.X[close_idx]
        out_data = np.stack((out1, out2), axis=1)
        out_data = torch.tensor(out_data, dtype=torch.float32).to(self.device)
        return out_data
    
    def _create_unsupervised_subdir(self):
        super()._create_output_subdir()
        self.model_dir = self.model_dir + 'contrastive_learning/'
        self.pred_dir = self.pred_dir + 'contrastive_learning/'
        self.embedding_dir = self.embedding_dir + 'contrastive_learning'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.pred_dir):
            os.makedirs(self.pred_dir)
        if not os.path.exists(self.embedding_dir):
            os.makedirs(self.embedding_dir)
    
    def _initialize_unsupervised_model(self, input_dim, epochs):
        self.epochs = epochs
        if self.unsupervised_type == 'autoencoder':
            dims = [input_dim, input_dim // 2, 1024, 512, 256, 128]
            self.unsupervised_model = Autoencoder(dims)
            self.unsupervised_criterion = nn.MSELoss()

        elif self.unsupervised_type != 'autoencoder':
            self.unsupervised_model = EncoderProjectionHead(input_dim)
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.unsupervised_model.parameters(), lr=0.001)
        self.unsupervised_model.to(self.device)
    
    def _train_unsupervised_model(self, train_dataset):
        X_train = train_dataset.X
        if self.unsupervised_type == 'autoencoder':
            X_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
            dataset = TensorDataset(X_tensor, X_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            self.unsupervised_model.train()
            for epoch in range(self.epochs):
                total_loss = 0
                for X_batch, _ in dataloader:
                    X_batch = X_batch.to(self.device)
                    self.optimizer.zero_grad()
                    reconstructed = self.unsupervised_model(X_batch)
                    loss = self.unsupervised_criterion(reconstructed, X_batch)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(dataloader)}")

        else:
            out_data = self._load_simclr_data(train_dataset)
            self.batch_size = out_data.shape[0]
            dataloader = DataLoader(out_data, batch_size=self.batch_size, shuffle=True)
            self.unsupervised_model.train()
            for epoch in range(self.epochs):
                for d in dataloader:
                    d = d.to(self.device).view(-1, d.size()[-1])
                    self.optimizer.zero_grad()
                    z = self.unsupervised_model(d)
                    logits, labels = info_nce_loss(z, self.batch_size, self.device)
                    loss = self.criterion(logits, labels)
                    loss.backward()
                    self.optimizer.step()
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item()}")


    def _save_result(self, result, save_dir, filename):
        with open(save_dir + filename + '.json', 'w') as f:
            result_to_save = {
                'Seed': self.seed,
                'Fold': filename.split('_')[1],
                'Model Type': self.model_type,
                'Model Hparams': self.model_hparams,
                'Unsupervised Type': self.unsupervised_type,  
                'Train Hparams': self.train_hparams,
                'Mixup Hparams': self.mixup_hparams,
                'Test Labels': result['test_labels'],
                'Test Logits': result['test_logits'],
                'Time Elapsed': result['time_elapsed']
            }
            if self.model_type not in ['rf', 'svm', 'rg']:
                result_to_save['Epoch Val Labels'] = result['epoch_val_labels']
                result_to_save['Epoch Val Logits'] = result['epoch_val_logits']
            json.dump(result_to_save, f, indent=4)


    def run_contrastive_learning(self, dataset, target, unsupervised_type, model_type, epochs, *args, **kwargs):
        self.filepaths = {
            'data': f'../data/{dataset}/data.tsv.xz',
            'meta': f'../data/{dataset}/meta.tsv',
            'target': f'../data/{dataset}/{target}.py',
            'unifrac map': f'../data/{dataset}/unifrac_map.pkl',
            'tree': '../data/WoL2/taxonomy.nwk' if not dataset.endswith('_ncbi') else '../data/WoL2/taxonomy_ncbi.nwk',
            'phylogeny tree': '../data/WoL2/phylogeny.nwk'
        }

        # Load tree
        self._load_tree(self.filepaths['tree'])

        # Load phylogeny tree
        self._load_tree(self.filepaths['phylogeny tree'], phylogeny=True)
        # Load data
        self._load_data(self.filepaths['data'], self.filepaths['meta'], self.filepaths['target'], self.filepaths['unifrac map'], preprocess=False)
        # Create output directory
        self.unsupervised_type = unsupervised_type
        self.model_type = model_type
        if self.data.clr_transformed != True:
            self.data.clr_transform()

        self.unsupervised_type = unsupervised_type
        self._create_unsupervised_subdir()
        
        input_dim = self.data.X.shape[-1]
        self._initialize_unsupervised_model(input_dim, epochs)
        self._train_unsupervised_model(self.data)

        # Here you can continue with any downstream tasks, e.g., classification or regression
        self._train()

def contrastive_learning_pipeline(dataset, target, unsupervised_type, model_type, seed, epochs):
    pipeline = ContrastiveLearningPipeline(seed, dataset)
    pipeline.run_contrastive_learning(dataset, target, unsupervised_type, model_type, epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, help='Random seed.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use for contrastive learning.')
    parser.add_argument("--target", type=str, required=True, help="Target to predict.")
    parser.add_argument("--unsupervised_type", type=str, required=True, choices=['autoencoder', 'vanilla', 'phylomix', 'compositional_cutmix', 'TADA', 'unifrac'], help="self-supervised model to use")
    parser.add_argument("--model_type", type=str, choices=['svm', 'rf', 'logisticCV', 'mlp'], default='logisticCV')
    parser.add_argument("--epochs", type=int, default=3000, help="Number of epochs for contrastive learning.")
    args = parser.parse_args()

    contrastive_learning_pipeline(args.dataset, args.target, args.unsupervised_type, args.model_type, args.seed, args.epochs)
