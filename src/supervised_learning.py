import argparse
import json
import os
import pickle
import time
from datetime import datetime
import copy
import matplotlib.pyplot as plt
# import umap
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from tqdm import tqdm
import sys
import atexit
import gc

import numpy as np
import torch
import biom
import torch.nn as nn
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import Timer, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV, LinearRegression, LogisticRegression
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAUROC, MulticlassAveragePrecision

from model import MLP, PopPhyCNN, TaxoNN
from mixup.data import PhylogenyDataset
from mixup import Mixup
from miostone import MIOSTONEModel
from pipeline import Pipeline

from io import StringIO
from skbio import TreeNode
from skbio.diversity.beta import weighted_unifrac


class DataModule(LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size, num_workers):  
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers 

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
class Classifier(LightningModule):
    def __init__(self, model, class_weight, metrics):
        super().__init__()
        self.model = model
        self.train_criterion = nn.CrossEntropyLoss(weight=class_weight)
        self.val_criterion = nn.CrossEntropyLoss()
        self.metrics = metrics
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Initialize lists to store logits and labels
        self.epoch_val_logits = []
        self.epoch_val_labels = []
        self.test_logits = None
        self.test_labels = None
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        if X.size()[0] > 1:
            logits = self.model(X)
            loss = self.train_criterion(logits, y)
            l0_reg = self.model.get_total_l0_reg()
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
    
            return loss + l0_reg
        else:
            return Nonel

    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        loss = self.val_criterion(logits, y)
        l0_reg = self.model.get_total_l0_reg()
        self.validation_step_outputs.append({'logits': logits, 'labels': y})
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=False)

        # Calculate metrics
        self.metrics.to(logits.device)
        scores = self.metrics(logits, y)
        for key, value in scores.items():
            self.log(f'val_{key}', value, on_step=False, on_epoch=True, prog_bar=False, logger=False)

        return loss + l0_reg
    
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        logits = torch.cat([x['logits'] for x in outputs], dim=0)
        labels = torch.cat([x['labels'] for x in outputs], dim=0)

        # Store logits and labels
        self.epoch_val_logits.append(logits.detach().cpu().numpy().tolist())
        self.epoch_val_labels.append(labels.detach().cpu().numpy().tolist())

        # Reset validation step outputs
        self.validation_step_outputs = []

    def test_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        self.test_step_outputs.append({'logits': logits, 'labels': y})
    
    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        logits = torch.cat([x['logits'] for x in outputs], dim=0)
        labels = torch.cat([x['labels'] for x in outputs], dim=0)

        # Store logits and labels
        self.test_logits = logits.detach().cpu().numpy().tolist()
        self.test_labels = labels.detach().cpu().numpy().tolist()

        # Reset test step outputs
        self.test_step_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [scheduler]

class TrainingPipeline(Pipeline):
    def __init__(self, seed):
        super().__init__(seed)
        self.model_type = None
        self.model_hparams = {}
        self.train_hparams = {}
        self.mixup_hparams = {}
        self.output_subdir = None

    def _create_output_subdir(self):
        self.pred_dir = self.output_dir + 'predictions/'
        self.model_dir = self.output_dir + 'models/'
        self.embedding_dir = self.output_dir + 'embeddings/'
        self.augtime_dir = self.output_dir + 'augtime/'
        self.data_dir = self.output_dir + 'datas/'
        for dir in [self.pred_dir, self.model_dir, self.data_dir, self.augtime_dir]:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def _create_subset(self, data, indices, normalize=True, one_hot_encoding=True, clr=True):
        X_subset = data.X[indices]
        y_subset = data.y[indices]
        ids_subset = data.ids[indices]
        subset = PhylogenyDataset(X_subset, y_subset, ids_subset, data.features)

        for i, idx1 in enumerate(indices):
            for j, idx2 in enumerate(indices):
                subset.unifrac_distances_map[i, j] = data.unifrac_distances_map[idx1, idx2]

        if one_hot_encoding:
            subset.one_hot_encode()
        if normalize:
            subset.normalize()
        if clr:
            subset.clr_transform()
        return subset
    
    def _apply_mixup(self, train_dataset, seed):
        mixup_processor = Mixup(train_dataset, self.tree, self.phylogeny_tree)
        min_threshold = np.quantile(mixup_processor.distances[mixup_processor.distances != 0], self.mixup_hparams['q_interval'][0])
        max_threshold = np.quantile(mixup_processor.distances[mixup_processor.distances != 0], self.mixup_hparams['q_interval'][1])
        num_samples = int(train_dataset.X.shape[0] * self.mixup_hparams['augment_ratio'])
        if self.aug_type in ['vanilla', 'aitchison', 'phylomix']:
            augmented_dataset = mixup_processor.mixup(min_threshold, max_threshold, num_samples, method=self.mixup_hparams['method'], alpha=self.mixup_hparams['alpha'], tree=self.mixup_hparams['tree'])
        elif self.aug_type == 'general_mixup':
            augmented_dataset = mixup_processor.general_mixup(min_threshold, max_threshold, num_samples)
        elif self.aug_type in ['intra_vanilla', 'intra_aitchison']:
            augmented_dataset = mixup_processor.intra_mixup(min_threshold, max_threshold, self.mixup_hparams['method'], num_samples, alpha=self.mixup_hparams['alpha'])
        elif self.aug_type == 'compositional_cutmix':
            augmented_dataset = mixup_processor.compositional_cutmix(min_threshold, max_threshold, num_samples)
        elif self.aug_type == 'intra_cutmix':
            augmented_dataset = mixup_processor.intra_cutmix(min_threshold, max_threshold, num_samples, height=self.mixup_hparams['depth'], num_subtrees=self.mixup_hparams['num_subtrees'], phylogeny_tree=self.phylogeny_tree)
        return augmented_dataset

    def _create_classifier(self, train_dataset, metrics):
        in_features = train_dataset.X.shape[1]
        out_features = train_dataset.num_classes
        class_weight = train_dataset.class_weight if self.train_hparams['class_weight'] == 'balanced' else [1] * out_features

        if self.model_type in ['rf', 'svm', 'linear', 'logisticCV']:
            class_weight = {key: value for key, value in enumerate(class_weight)}
            if self.model_type == 'rf' and self.aug_type in ['None', 'compositional_cutmix', 'TADA']:
                classifier = RandomForestClassifier(class_weight=class_weight, random_state=self.seed)
            elif self.model_type == 'rf' and self.aug_type in ['vanilla', 'phylomix']:
                classifier = RandomForestRegressor(random_state=self.seed)
            elif self.model_type == 'svm' and self.aug_type in ['None', 'compositional_cutmix', 'TADA']:
                classifier = SVC(kernel='linear', probability=True, class_weight=class_weight, random_state=self.seed)
            elif self.model_type == 'svm' and self.aug_type in ['vanilla', 'phylomix']:
                classifier = SVR(kernel='linear')
            elif self.model_type == 'linear' and self.aug_type in ['None', 'compositional_cutmix', 'TADA']:
                classifier = LogisticRegression(class_weight=class_weight, random_state=self.seed, penalty=None)
            elif self.model_type == 'linear' and self.aug_type in ['vanilla', 'phylomix']:
                classifier = LinearRegression()
            elif self.model_type == 'logisticCV':
                classifier = LogisticRegressionCV(class_weight=class_weight, penalty='l2', Cs=50)
        else:
            class_weight = torch.tensor(class_weight).float()
            if self.model_type == 'mlp':
                model = MLP(in_features, out_features, **self.model_hparams)
            elif self.model_type == 'taxonn':
                model = TaxoNN(self.tree, out_features, train_dataset, **self.model_hparams)
                print(len(list(self.tree.ete_tree.leaves())))
            elif self.model_type == 'popphycnn':
                model = PopPhyCNN(self.tree, out_features, **self.model_hparams)
            elif self.model_type == 'miostone':
                model = MIOSTONEModel(self.tree, out_features, **self.model_hparams)
            else:
                raise ValueError(f"Invalid model type: {self.model_type}")
    
            classifier = Classifier(model, class_weight, metrics)

        return classifier

    def _run_sklearn_training(self, classifier, train_dataset, test_dataset):
        time_elapsed = 0
        if self.train_hparams['max_epochs'] > 0:
            start_time = time.time()
            classifier.fit(train_dataset.X, train_dataset.y)
            time_elapsed = time.time() - start_time

        test_labels = test_dataset.y.tolist()
        test_logits = classifier.predict_proba(test_dataset.X).tolist()

        return {
            'test_labels': test_labels,
            'test_logits': test_logits,
            'time_elapsed': time_elapsed,
        }
    
    def _run_regression_training(self, classifier, train_dataset, test_dataset):
        time_elapsed = 0
        if self.aug_type in ['vanilla', 'phylomix']:
            y_label = train_dataset.y[:, 0]
        else:
            y_label = train_dataset.y
        if self.train_hparams['max_epochs'] > 0:
            start_time = time.time()
            classifier.fit(train_dataset.X, y_label)
            time_elapsed = time.time() - start_time

        test_labels = test_dataset.y.tolist()
        prob1 = classifier.predict(test_dataset.X)
        prob2 = 1 - prob1
        test_logits = np.column_stack((prob1, prob2)).tolist()

        return {
            'test_labels': test_labels,
            'test_logits': test_logits,
            'time_elapsed': time_elapsed,
        }

    def _run_pytorch_training(self, classifier, train_dataset, val_dataset, test_dataset, filename):
        timer = Timer()
        logger = TensorBoardLogger(self.output_dir + 'logs/', name=filename)
        data_module = DataModule(train_dataset, val_dataset, test_dataset, batch_size=self.train_hparams['batch_size'], num_workers=1)

        trainer = Trainer(
            max_epochs=self.train_hparams['max_epochs'],
            enable_progress_bar=True, 
            enable_model_summary=False,
            enable_checkpointing=False,
            log_every_n_steps=1,
            logger=logger,
            callbacks=[timer],
            accelerator='gpu',
            devices=1,
            deterministic=True
        )
        trainer.fit(classifier, datamodule=data_module)
        trainer.test(classifier, datamodule=data_module)

        return {
            'epoch_val_labels': classifier.epoch_val_labels,
            'epoch_val_logits': classifier.epoch_val_logits,
            'test_labels': classifier.test_labels,
            'test_logits': classifier.test_logits,
            'time_elapsed': timer.time_elapsed('train')
        }

    def _run_training(self, classifier, train_dataset, val_dataset, test_dataset, filename):
        if self.model_type in ['rf', 'svm', 'linear', 'logisticCV']:
            if self.aug_type in ['vanilla', 'phylomix']:
                return self._run_regression_training(classifier, train_dataset, test_dataset)
            else:
                return self._run_sklearn_training(classifier, train_dataset, test_dataset)
        else:
            return self._run_pytorch_training(classifier, train_dataset, val_dataset, test_dataset, filename)

    
    def _save_model(self, classifier, save_dir, filename):
        if self.model_type in ['rf', 'svm', 'linear']:
            pickle.dump(classifier, open(save_dir + filename + '.pkl', 'wb'))
        else:
            torch.save(classifier.model.state_dict(), save_dir + filename + '.pt')
    
    def _save_result(self, result, save_dir, filename):
        with open(save_dir + filename + '.json', 'w') as f:
            result_to_save = {
                'Seed': self.seed,
                'Fold': filename.split('_')[1],
                'Model Type': self.model_type,
                'Model Hparams': self.model_hparams,
                'Augmentation Type': self.aug_type,
                'Train Hparams': self.train_hparams,
                'Mixup Hparams': self.mixup_hparams,
                'Test Labels': result['test_labels'],
                'Test Logits': result['test_logits'],
                'Time Elapsed': result['time_elapsed']
            }
            if self.model_type not in ['rf', 'svm', 'linear']:
                result_to_save['Epoch Val Labels'] = result['epoch_val_labels']
                result_to_save['Epoch Val Logits'] = result['epoch_val_logits']
            json.dump(result_to_save, f, indent=4)

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
        
        # Define cross-validation strategy
        kf = KFold(n_splits=self.train_hparams['k_folds'], shuffle=True, random_state=self.seed)

        # Training loop
        fold_test_labels = []
        fold_test_logits = []
        for fold, (train_index, test_index) in enumerate(kf.split(self.data.X, self.data.y)):
            normalize = False
            clr = False if self.model_type in ['popphycnn'] or self.aug_type in ['TADA', 'phylomix'] else True
            one_hot_encoding = False if self.aug_type in ['TADA', 'compositional_cutmix', 'None'] else True
            train_dataset = self._create_subset(self.data, 
                                                train_index, 
                                                normalize=normalize,
                                                one_hot_encoding=False,
                                                clr = clr) 
            test_dataset = self._create_subset(self.data, 
                                            test_index,
                                            normalize=normalize,
                                            one_hot_encoding=False, 
                                            clr=clr if not self.aug_type in ['TADA', 'phylomix'] \
                                            or self.model_type == 'popphycnn' else True)
            
            train_dataset = train_dataset.subsample(self.train_hparams['subsample_ratio'], seed=self.seed)
            if one_hot_encoding:
                train_dataset.one_hot_encode()
            # Apply mixup if specified
            if self.aug_type == 'TADA':
                # Number of augmented samples per original sample
                augmentation_factor = 3

                # Calculate the indices for the augmented samples
                augmented_indices = []
                for index in train_index:
                    start_index = index * augmentation_factor
                    augmented_indices.extend(range(start_index, start_index + augmentation_factor))
                
                max_ind = self.TADA_aug.shape[0] - 1
                aug_label = np.repeat(train_dataset.y, 3, axis = 0)
                augmented_indices = [ind for ind, y in zip(augmented_indices, aug_label) if ind <= max_ind ]
                aug_label = [y for ind, y in zip(augmented_indices, aug_label) if ind <= max_ind ]
                train_dataset.y = np.concatenate((train_dataset.y, aug_label))
                train_dataset.X = np.vstack((train_dataset.X, self.TADA_aug[augmented_indices, :]))
                if self.model_type != 'popphycnn':
                    train_dataset.clr_transform()
                augmented_dataset = train_dataset


            if self.aug_type != 'TADA':
                augmented_dataset = self._apply_mixup(train_dataset, self.seed)
                if self.aug_type == 'phylomix' and self.model_type != 'popphycnn':
                    augmented_dataset.clr_transform()
            else:
                augmented_dataset = train_dataset
            classifier = self._create_classifier(augmented_dataset, metrics)
            # self.model = classifier.model

            # Convert to tree matrix if specified and apply standardization
            if self.model_type == 'popphycnn':
                scaler = augmented_dataset.to_popphycnn_matrix(self.tree)
                test_dataset.to_popphycnn_matrix(self.tree, scaler=scaler)

            # Set filename
            filename = f"{self.seed}_{fold}_{self.model_type}_{self.aug_type}"
            if self.model_type == 'miostone':
                filename += f"_{self.model_hparams['node_gate_type']}{self.model_hparams['node_gate_param']}"
                filename += f"_{self.model_hparams['node_dim_func']}{self.model_hparams['node_dim_func_param']}"
                filename += f"_{self.model_hparams['prune_mode']}"
            if self.mixup_hparams:
                filename += f"_sampleratio{self.mixup_hparams['augment_ratio']}"
                filename += f"_alpha{self.mixup_hparams['alpha']}"
                filename += f"_l{self.mixup_hparams['q_interval'][0]}"
                filename += f"_u{self.mixup_hparams['q_interval'][1]}"
                filename += f"_depth{self.mixup_hparams['depth']}"
                filename += f"_numsubtrees{self.mixup_hparams['num_subtrees']}"
            if "subsample_ratio" in self.train_hparams:
                filename += f"_subsampe{self.train_hparams['subsample_ratio']}"
            if "percent_features" in self.train_hparams:
                filename += f"_p{self.train_hparams['percent_features']}"
            if "num_frozen_layers" in self.train_hparams:
                filename += f"_frozen{self.train_hparams['num_frozen_layers']}"
            if "pretrain_num_epochs" in self.train_hparams:
                filename += f"_pretrain{self.train_hparams['pretrain_num_epochs']}"
                if self.train_hparams['max_epochs'] == 0:
                    filename += "_zs"
            filename += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            result = self._run_training(classifier, augmented_dataset, test_dataset, test_dataset, filename)
            fold_test_labels.append(torch.tensor(result['test_labels']))
            fold_test_logits.append(torch.tensor(result['test_logits']))
                     
            # Save results and model
            self._save_result(result, self.pred_dir, filename)
            self._save_model(classifier, self.model_dir, filename)

        # Calculate metrics
        test_labels = torch.cat(fold_test_labels, dim=0)
        test_logits = torch.cat(fold_test_logits, dim=0)
        metrics.to(test_labels.device)
        test_scores = metrics(test_logits, test_labels)
        print(f"Test scores:")
        for key, value in test_scores.items():
            print(f"{key}: {value.item()}")

    def run(self, dataset, target, model_type, aug_type, augment_ratio, lower_bound, upper_bound, alpha, 
            subsample_ratio, *args, **kwargs):
        # Define filepaths
        self.filepaths = {
            'data': f'../data/{dataset}/data.tsv.xz',
            'meta': f'../data/{dataset}/meta.tsv',
            'target': f'../data/{dataset}/{target}.py',
            'unifrac map': f'../data/{dataset}/unifrac_map.pkl',
            'tree': '../data/WoL2/taxonomy.nwk' if not dataset.endswith('_ncbi') else '../data/WoL2/taxonomy_ncbi.nwk',
            'phylogeny tree': '../data/WoL2/phylogeny.nwk',
            'TADA': f'../../TADA/{dataset}_augmented/augmented_data.biom',
            'TADA1': f'../../TADA/{dataset}_augmented1/augmented_data.biom',
            'TADA2': f'../../TADA/{dataset}_augmented2/augmented_data.biom',
            'TADA_y': f'../../TADA/{dataset}_augmented/label.npy'
        }

        # Load tree
        self._load_tree(self.filepaths['tree'])

        # Load TADA augmented data
        table = biom.load_table(self.filepaths['TADA'])
        mat = table.matrix_data.todense()
        self.TADA_aug = np.array(mat.T, dtype=np.float32)

        table1, table2 = biom.load_table(self.filepaths['TADA1']), biom.load_table(self.filepaths['TADA2'])
        mat1, mat2 = table1.matrix_data.todense(), table2.matrix_data.todense()
        self.TADA_aug1, self.TADA_aug2 = np.array(mat1.T, dtype=np.float32), np.array(mat2.T, dtype=np.float32)

        # Load phylogeny tree
        self._load_tree(self.filepaths['phylogeny tree'], phylogeny=True)
        
        # Load data
        self._load_data(self.filepaths['data'], self.filepaths['meta'], self.filepaths['target'], self.filepaths['unifrac map'], preprocess=False)
        # Create output directory
        self._create_output_subdir()

        # Configure default model parameters
        self.model_type = model_type
        self.aug_type = aug_type
        self.augment_ratio = str(augment_ratio)
        if self.model_type == 'miostone':
            self.model_hparams['node_min_dim'] = 1
            self.model_hparams['node_dim_func'] = 'linear'
            self.model_hparams['node_dim_func_param'] = 0.6
            self.model_hparams['node_gate_type'] = 'concrete'
            self.model_hparams['node_gate_param'] = 0.3
            self.model_hparams['prune_mode'] = 'taxonomy'
        elif self.model_type == 'popphycnn':
            self.model_hparams['num_kernel'] = 32
            self.model_hparams['kernel_height'] = 3
            self.model_hparams['kernel_width'] = 10
            self.model_hparams['num_fc_nodes'] = 512
            self.model_hparams['num_cnn_layers'] = 1
            self.model_hparams['num_fc_layers'] = 1
            self.model_hparams['dropout'] = 0.3

        # Configure default training parameters
        self.train_hparams['k_folds'] = 5
        self.train_hparams['batch_size'] = 512
        self.train_hparams['max_epochs'] = 200
        self.train_hparams['class_weight'] = 'balanced'
        self.train_hparams['subsample_ratio'] = subsample_ratio

        # Configure default mixup parameters
        if aug_type != 'None' and aug_type != 'None0':
            self.mixup_hparams['method'] = aug_type
            self.mixup_hparams['augment_ratio'] = augment_ratio
            self.mixup_hparams['q_interval'] = (lower_bound, upper_bound)
            self.mixup_hparams['alpha'] = alpha
            self.mixup_hparams['tree'] = 'phylogeny'
            if len(kwargs) > 0:
                for k, v in kwargs.items():
                    self.mixup_hparams[k] = v
        
        # Train the model
        self._train()

def run_training_pipeline(dataset, target, model_type, aug_type, seed, augment_ratio, lower_bound, upper_bound, alpha, subsample_ratio,
                          *args, **kwargs):
    pipeline = TrainingPipeline(seed=seed)
    pipeline.run(dataset, target, model_type, aug_type, augment_ratio, lower_bound, upper_bound, alpha, subsample_ratio, *args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Random seed.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use.")
    parser.add_argument("--target", type=str, required=True, help="Target to predict.")
    parser.add_argument("--model_type", type=str, required=True, 
                        choices=['rf', 'svm', 'mlp', 'taxonn', 'popphycnn', 'miostone', 'linear', 'logisticCV'],
                        help="Model type to use.")
    parser.add_argument("--augmentation_type", type=str, 
                        choices=['vanilla', 'aitchison', 'cutmix', 'None', 'phylomix', 'compositional_cutmix', 'TADA',
                                 'general mixup', 'intra_vanilla', 'intra_aitchison'], default='None')
    parser.add_argument("--augment_ratio", type=int, default=3)
    parser.add_argument("--lower_bound", type=float, default=0.0)
    parser.add_argument("--upper_bound", type=float, default=1.0)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--num_subtrees", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=2)
    parser.add_argument("--subsample_ratio", type=float, default=1.0)
    args = parser.parse_args()

    run_training_pipeline(args.dataset, args.target, args.model_type, args.augmentation_type, args.seed, args.augment_ratio,
                          args.lower_bound, args.upper_bound, args.alpha, args.subsample_ratio,
                            depth=args.depth, num_subtrees=args.num_subtrees)
