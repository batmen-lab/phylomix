import argparse
import glob
import json
import os
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest
from matplotlib import pyplot as plt
from torchmetrics import MetricCollection
from torchmetrics.classification import (MulticlassAccuracy, MulticlassAUROC,
                                         MulticlassAveragePrecision, MulticlassROC)


class ResultsAnalyzer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.results = None
        self.dataset = None
        self.target = None
        self.model_type_labels = {
            'taxonn': 'TaxoNN',
            'popphycnn': 'PopPhy-CNN',
            'mlp': 'Multilayer Perceptron',
            'svm': 'Support Vector Machine',
            'rf': 'Random Forest',
            'miostone': 'MIOSTONE'
        }
        self.model_type_palette = {
            'phylomix': '#ffd700',
            'None': '#5da5da',
            'TADA': '#3498db',
            'compositional_cutmix': '#1f77b4',
            'vanilla': '#aec7e8',
            'autoencoder': '#5da5da',
            'unifrac': '#7eaedc'
        }
        self.dataset_labels = {
            'rumc_pd': 'RUMC - PD',
            'alzbiom_ad': 'AlzBiom - AD',
            'ibd200_type': 'IBD200 - Type',
            'asd_stage': 'ASD - Stage',
            'tbc_pd': 'TBC - PD',
            'hmp2_type': 'HMP2 - Type',
            'gd_cohort': 'GD - Cohort'
        }
        
    def _load_results(self, dataset, target, transfer_learning=False, contrastive_learning=True):
        self.dataset = dataset
        self.target = target

        results = []
        dir_name = 'predictions/'
        if transfer_learning:
            dir_name += 'transfer_learning/'
        if contrastive_learning:
            dir_name += 'contrastive_learning/'
        results_dir = os.path.join(self.output_dir, dataset, target, dir_name)
        for filepath in glob.glob(os.path.join(results_dir, '*.json')):
            with open(filepath, 'r') as f:
                result = json.load(f)
                if result['Mixup Hparams']:
                    result['Mixup Hparams']['q_interval'] = 1
                results.append(result)

        self.results = pd.DataFrame(results)
        for col in ['Model Hparams', 'Train Hparams', 'Mixup Hparams']:
            if col == 'Mixup Hparams':
                self.results[col] = self.results[col].apply(lambda x: frozenset(x.items()))
            else:
                self.results[col] = self.results[col].apply(lambda x: frozenset())
        for col in ['Epoch Val Labels', 'Epoch Val Logits']:
            self.results[col] = [np.array([0]) for _ in range(len(self.results))]
        for col in ['Test Labels', 'Test Logits']:
            self.results[col] = self.results[col].apply(lambda x: np.array(x))

    def _plot_roc(self):
        models = np.unique(self.results['Model Type'].values).tolist()
        augmentations = np.unique(self.results['Augmentation Type'].values).tolist()

        for model in models:
            for aug in augmentations:
                result_to_plot = self.results[(self.results['Model Type'] == model) & (self.results['Augmentation Type'] == aug)]
                test_logits = torch.tensor(np.concatenate(result_to_plot['Test Logits'].values))
                test_labels = torch.tensor(np.concatenate(result_to_plot['Test Labels'].values))

                metric = MulticlassROC(num_classes=self.results['Test Logits'].iloc[0].shape[1])
                metric.update(test_logits, test_labels)
                fig, ax = metric.plot(score=True)
                ax.set_title(f'{model} {aug}')

                filename = f'../plots/roc/{model}_{aug}'
                fig.savefig(filename)

    def _compute_metrics(self, contrastive_learning=False):
        num_classes = self.results['Test Logits'].iloc[0].shape[1]
        metrics = MetricCollection([
            MulticlassAccuracy(num_classes=num_classes),
            MulticlassAUROC(num_classes=num_classes),
            MulticlassAveragePrecision(num_classes=num_classes),
        ])

        if not contrastive_learning:
            grouped_results = self.results.groupby(['Seed', 'Model Type', 'Augmentation Type', 'Model Hparams', 'Train Hparams', 'Mixup Hparams'])
        else:
            grouped_results = self.results.groupby(['Seed', 'Model Type', 'Unsupervised Type', 'Model Hparams', 'Train Hparams', 'Mixup Hparams'])

        concatenated_test_labels = {}
        concatenated_test_logits = {}

        for name, group in grouped_results:
            concatenated_test_labels[name] = np.concatenate(group['Test Labels'].values)
            concatenated_test_logits[name] = np.concatenate(group['Test Logits'].values)

        rows = []
        for (seed, model_type, t, model_hparams, train_hparams, mixup_hparams), name in zip(grouped_results.groups.keys(), grouped_results.groups):
            test_labels = torch.tensor(concatenated_test_labels[name])
            test_logits = torch.tensor(concatenated_test_logits[name])
            test_scores = metrics(test_logits, test_labels)
            row = {
                'Seed': seed,
                'Model Type': model_type,
                'Augmentation Type': t if not contrastive_learning else None,
                'Unsupervised Type': t if contrastive_learning else None,
                'test Accuracy': test_scores['MulticlassAccuracy'].item(),
                'test AUROC': test_scores['MulticlassAUROC'].item(),
                'test AUPRC': test_scores['MulticlassAveragePrecision'].item(),
            }
            for key, value in model_hparams:
                row[key] = value
            for key, value in train_hparams:
                row[key] = value
            for key, value in mixup_hparams:
                row[key] = value
            rows.append(row)
        
        self.scores = pd.DataFrame(rows)

    def _visualize_val_curves(self):
        rows = []
        for i in range(len(self.scores)):
            original_row = self.scores.iloc[i]
            if original_row['epoch val Accuracy'] is not None:
                for epoch in range(len(original_row['epoch val Accuracy'])):
                    row = {
                        'Seed': original_row['Seed'],
                        'Model Type': original_row['Model Type'],
                        'Augmentation Type': original_row['Augmentation Type'],
                        'epoch': epoch,
                        'epoch val Accuracy': original_row['epoch val Accuracy'][epoch],
                        'epoch val AUROC': original_row['epoch val AUROC'][epoch],
                        'epoch val AUPRC': original_row['epoch val AUPRC'][epoch],
                    }
                    rows.append(row)
        
        curves = pd.DataFrame(rows)

        sns.set(style="whitegrid")

        metrics = ['Accuracy', 'AUROC', 'AUPRC']
        augmentations = np.unique(self.results['Augmentation Type'].values).tolist()
        models = np.unique(self.results['Model Type'].values).tolist()
        models = [model for model in models if model not in ['rf', 'svm', 'linear', 'logisticCV']]

        fig, axes = plt.subplots(len(models), len(metrics), figsize=(15, 10))

        fig.suptitle(f'{self.dataset} - {self.target}', fontsize=20)
        for k, model in enumerate(models):
            curves_to_plot = curves[curves['Model Type'] == model]
            for i, metric in enumerate(metrics):
                sns.lineplot(data=curves_to_plot,
                             x='epoch',
                             y='epoch val ' + metric,
                             hue='Augmentation Type',
                             hue_order=augmentations,
                             errorbar="sd",
                             ax=axes[k][i])
            
                axes[k][i].set_title(f'{metric}')
                axes[k][i].set_xlabel('Epoch')
                axes[k][i].set_ylabel(f'{model}')

        plt.tight_layout()
        plt.show()

    def _visualize_alpha(self):
        sns.set_theme(style="white")
        metrics = ['test Accuracy', 'test AUROC', 'test AUPRC']
        models = np.unique(self.results['Model Type'].values).tolist()

        self.scores['sample_ratio'].fillna(0.0, inplace=True)
        self.scores['alpha'].fillna(0.0, inplace=True)
        self.scores['tree'].fillna(0.0, inplace=True)
        self.scores = self.scores[(self.scores['sample_ratio'] == 3.0) | (self.scores['sample_ratio'] == 0.0)]
        self.scores = self.scores[(self.scores['tree'] == 'phylogeny') | (self.scores['tree'] == 0.0)]
        
        self.scores['alpha'] = self.scores['alpha'].astype(str)
        self.scores['alpha'] = self.scores['alpha'].map({
            '0.0': 'None',
            '0.5': 'Beta(0.5, 0.5)',
            '1.0': 'Beta(1.0, 1.0)',
            '2.0': 'Beta(2.0, 2.0)'
        }).fillna(self.scores['alpha'])

        order_alpha = ['None', 'Beta(0.5, 0.5)', 'Beta(1.0, 1.0)', 'Beta(2.0, 2.0)']
        self.scores['alpha'] = pd.Categorical(self.scores['alpha'], categories=order_alpha, ordered=True)

        order_model_type = ['linear', 'svm', 'rf', 'mlp', 'miostone']
        self.scores['Model Type'] = pd.Categorical(self.scores['Model Type'], categories=order_model_type, ordered=True)

        for metric in metrics:
            plt.figure(figsize=(24, 12))

            ax = sns.barplot(data=self.scores,
                             x='Model Type',
                             y=metric,
                             hue='alpha',
                             dodge=True,
                             errwidth=2,
                             palette="Blues")

            plt.title(f'Comparison of {metric.replace("test ", "")} across Models and Alpha Values', fontsize=22)
            plt.xlabel('Model Type', fontsize=18)
            plt.ylabel(metric, fontsize=18)
            
            plt.xticks(rotation=45, fontsize=12)
            plt.ylim(0.5, 1.05)

            self._annotate_bar_plot(ax)
            
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            directory = f'../plots/{self.dataset}'
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = f'{metric}_alpha.pdf'
            full_path = os.path.join(directory, filename)
            plt.savefig(full_path)
            plt.show()

    def _visualize_sample_ratio(self):
        sns.set_theme(style="white")
        metrics = ['test Accuracy', 'test AUROC', 'test AUPRC']
        models = np.unique(self.results['Model Type'].values).tolist()

        self.scores['sample_ratio'].fillna(0.0, inplace=True)
        self.scores['alpha'].fillna(0.0, inplace=True)
        self.scores['tree'].fillna(0.0, inplace=True)
        self.scores = self.scores[(self.scores['alpha'] == 2.0) | (self.scores['alpha'] == 0.0)]
        self.scores = self.scores[(self.scores['tree'] == 'phylogeny') | (self.scores['tree'] == 0.0)]

        self.scores['sample_ratio'] = self.scores['sample_ratio'].astype(str)
        self.scores['sample_ratio'] = self.scores['sample_ratio'].map({
            '0.0': 'None',
            '1.0': 'cutmix-1',
            '3.0': 'cutmix-3',
            '5.0': 'cutmix-5',
            '10.0': 'cutmix-10'
        }).fillna(self.scores['sample_ratio'])

        order = ['None', 'cutmix-1', 'cutmix-3', 'cutmix-5', 'cutmix-10']
        self.scores['sample_ratio'] = pd.Categorical(self.scores['sample_ratio'], categories=order, ordered=True)

        order_model_type = ['linear', 'svm', 'rf', 'mlp', 'miostone']
        self.scores['Model Type'] = pd.Categorical(self.scores['Model Type'], categories=order_model_type, ordered=True)

        for metric in metrics:
            plt.figure(figsize=(28, 12))

            ax = sns.barplot(data=self.scores,
                             x='Model Type',
                             y=metric,
                             hue='sample_ratio',
                             dodge=True,
                             errwidth=2,
                             palette="Blues")

            plt.title(f'Comparison of {metric.replace("test ", "")} across Models and Sample Ratios', fontsize=22)
            plt.xlabel('Model Type', fontsize=18)
            plt.ylabel(metric, fontsize=18)

            plt.xticks(rotation=45, fontsize=12)
            plt.ylim(0.5, 1.05)

            self._annotate_bar_plot(ax)

            handles, labels = ax.get_legend_handles_labels()
            if labels:
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            directory = f'../plots/{self.dataset}'
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = f'{metric}_sample_ratio.pdf'
            full_path = os.path.join(directory, filename)
            plt.savefig(full_path)
            plt.show()

    def _visualize_test_scores(self, deeplearning=True, contrastive_learning=False):
        sns.set_theme(style="white")
        metrics = ['test Accuracy', 'test AUROC', 'test AUPRC']
        models = np.unique(self.results['Model Type'].values).tolist()

        if not contrastive_learning:
            extra_type = np.unique(self.results['Augmentation Type'].values).tolist()
            order = 'Augmentation Type'
            extra_type = ['None', 'vanilla', 'compositional_cutmix', 'TADA', 'phylomix']
        else:
            extra_type = np.unique(self.results['Unsupervised Type'].values).tolist()   
            order = 'Unsupervised Type' 
            extra_type = ['autoencoder', 'unifrac', 'vanilla', 'compositional_cutmix', 'TADA', 'phylomix']
        if deeplearning:
            models = [model for model in models if model not in ['rf', 'svm', 'linear', 'logisticCV']]
            augmentations = [aug for aug in extra_type if aug not in ['intra_vanilla', 'intra_aitchison']]
        else:
            models = [model for model in models if model in ['rf', 'svm', 'linear', 'logisticCV']]

        for model in models:
            model_to_plot = self.scores[(self.scores['Model Type'] == model)]
            for metric in metrics:
                plt.figure(figsize=(12, 10))
                ax = sns.barplot(data=model_to_plot,
                                 x=order,
                                 order=extra_type,
                                 y=metric, 
                                 hue=order,
                                 palette=self.model_type_palette,
                                 dodge=False,
                                 errwidth=2)
                
                plt.title(f'{model}-{metric.replace("test ", "")}', fontsize=22)
                plt.xlabel('')
                plt.ylabel('')

                plt.xticks([i for i in range(len(extra_type))], extra_type, fontsize=12)
                plt.ylim(0.5, 1.05)

                self._annotate_bar_plot(plt.gca())
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                def ttest_rel_less(group_data1, group_data2, **stats_params):
                    return stats.ttest_rel(group_data1, group_data2, alternative='less', **stats_params)
                
                custom_long_name = 't-test paired samples (less)'
                custom_short_name = 't-test_rel_greater'
                custom_func = ttest_rel_less
                custom_stat_test = StatTest(custom_func, custom_long_name, custom_short_name)
                
                pairs1 = [('None', 'phylomix'), ('TADA', 'phylomix'), ('compositional_cutmix', 'phylomix'), ('vanilla', 'phylomix')]
                pairs_contrastive_learning = [('autoencoder', 'phylomix'), ('compositional_cutmix', 'phylomix'), ('vanilla', 'phylomix'), ('TADA', 'phylomix'), ('unifrac', 'phylomix')]
                if not contrastive_learning:
                    annotator = Annotator(ax, pairs1, data=model_to_plot, x='Augmentation Type', y=metric, order=extra_type)
                else:
                    annotator = Annotator(ax, pairs_contrastive_learning, data=model_to_plot, x='Unsupervised Type', y=metric, order=extra_type)
                annotator.configure(test=custom_stat_test, text_format='full', show_test_name=False)
                annotator.apply_and_annotate()

                plt.tight_layout()

                directory = f'../plots/{self.dataset}'
                if contrastive_learning:
                    directory = f'../plots_contrastive_learning/{self.dataset}'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filename = f'{metric.replace("test ", "")}_{model}.pdf'
                full_path = os.path.join(directory, filename)
                plt.savefig(full_path)
                print(full_path)
                plt.show()

    def _visualize_time_elapsed(self):
        time_dir = os.path.join(self.output_dir, self.dataset, self.target) + '/augtime/'
        sns.set_theme(style="white")

        with open(time_dir + 'timing.txt', 'r') as file:
            lines = file.readlines()

        times = [float(line.strip()) for line in lines]
        average_time = sum(times) / len(times) if times else 0

        models = ['linear', 'svm', 'rf', 'mlp', 'miostone']
        self.results = self.results[(self.results['Augmentation Type'] == 'None') | (self.results['Augmentation Type'] == 'phylomix')]
        self.results = self.results[self.results['Model Type'].isin(models)]
        self.results.loc[self.results['Augmentation Type'] == 'phylomix', 'Time Elapsed'] += average_time

        plt.figure(figsize=(30, 10))

        ax = sns.barplot(data=self.results,
                         x='Model Type',
                         y='Time Elapsed',
                         hue='Augmentation Type',
                         hue_order=['None', 'phylomix'],
                         order=models,
                         palette="Blues",
                         dodge=True,
                         errwidth=2)
        
        hue_order = ['None', 'phylomix']
        for i, bar in enumerate(ax.patches):
            if i > 4:
                current_height = bar.get_height()
                prev_height = current_height - average_time
                plt.bar(bar.get_x() + bar.get_width() / 2, average_time, bottom=prev_height, color="gold", width=bar.get_width())
            
        from matplotlib.patches import Patch
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Patch(color='gold', label='Augmentation Time'))
        plt.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5))

        self._annotate_bar_plot(ax)
        plt.title('Time Elapsed', fontsize=22)
        plt.xlabel('Model Type', fontsize=18)
        plt.ylabel('Total Running Time in Seconds (s)', fontsize=18)
        plt.xticks(rotation=45, fontsize=12)

        plt.tight_layout()

        directory = f'../plots_time/'
        if not os.exists(directory):
            os.makedirs(directory)
        filename = f'{self.dataset}_time_elapsed.pdf'
        full_path = os.path.join(directory, filename)
        plt.savefig(full_path)
        plt.show()

    def _annotate_bar_plot(self, ax):
        is_log_scale = ax.get_yscale() == 'log'

        ylim = ax.get_ylim()
        log_range = np.log10(ylim[1]/ylim[0]) if is_log_scale else None

        for p, err_bar in zip(ax.patches, ax.lines):
            bar_length = p.get_height()
            err_bar_length = err_bar.get_ydata()[1] - err_bar.get_ydata()[0]
            text_position = bar_length + err_bar_length / 2

            if is_log_scale:
                log_offset = 0.01 * log_range
                y = text_position * (10 ** log_offset)
            else:
                offset = 0.01 * (ylim[1] - ylim[0])
                y = text_position + offset

            x = p.get_x() + p.get_width() / 2
            position = (x, y)
            ax.annotate(f"{bar_length:.4f}", xy=position, ha='center', va='center', fontsize=15, color='black')

    def _visualize_test_scores_dot_plot(self, contrastive_learning):
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        sns.set_theme(style="white")
        all_summary_results = []
        model_types = self.scores['Model Type'].unique()
        t = 'Augmentation Type' if not contrastive_learning else 'Unsupervised Type'
        base = 'None' if not contrastive_learning else 'phylomix'
        metrics = ['test AUROC', 'test AUPRC']
        marker_styles = {'phylomix': 'o', 'TADA': 's', 'compositional_cutmix': 'X', 'vanilla': 'D', 'autoencoder': '^', 'unifrac': 'v'}
        
        for model in model_types:
            for metric in metrics:
                comparison_df = pd.DataFrame()

                for seed in self.scores['Seed'].unique():
                    model_seed_scores = self.scores[(self.scores['Seed'] == seed) & (self.scores['Model Type'] == model)]
                    none_score = model_seed_scores[model_seed_scores[t] == base][metric].values

                    if len(none_score) > 0:
                        for _, row in model_seed_scores.iterrows():
                            comparison_df = comparison_df._append({
                                'Seed': seed,
                                'Type': row[t],
                                'None Score': none_score[0],
                                f'{metric} Score': row[metric]
                            }, ignore_index=True)

                comparison_df['Marker Size'] = comparison_df['Type'].apply(lambda x: 300 if x == 'phylomix' else 50)

                summary_results = self.perform_statistical_tests(comparison_df, metric, model, metric)
                all_summary_results.extend(summary_results)
                plt.figure(figsize=(10, 8))

                sns.scatterplot(data=comparison_df, x='None Score', y=f'{metric} Score', hue='Type', style='Type', markers=marker_styles, palette=self.model_type_palette, size='Marker Size', sizes=(100, 200), legend=False)

                min_score = min(comparison_df['None Score'].min(), comparison_df[f'{metric} Score'].min())
                max_score = max(comparison_df['None Score'].max(), comparison_df[f'{metric} Score'].max())
                plt.xlim(min_score, max_score)
                plt.ylim(min_score, max_score)

                plt.plot([min_score, max_score], [min_score, max_score], color='gray', linestyle='--')

                plt.xlabel('Base score')
                plt.ylabel(f'Other {t} Score ({metric})')
                plt.title(f'{self.dataset} - {self.target} - phylomix ({metric})')

                handles, labels = [], []
                for key in marker_styles.keys():
                    if key not in ['autoencoder', 'unifrac']: 
                        handles.append(plt.Line2D([0], [0], marker=marker_styles[key], color='w', markerfacecolor=self.model_type_palette[key], markersize=10))
                        labels.append(key)
                plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=t)

                plt.tight_layout()

                dir = f'../plots_scatter/{self.dataset}'
                if not os.path.isdir(dir):
                    os.makedirs(dir)
                plot_filename = f"{dir}/{model}_{metric}_comparison.pdf"
                plt.savefig(plot_filename)
                plt.close()
        self.print_summary_results(all_summary_results, results_dir)

    def _visualize_subsample(self):
        sns.set_theme(style="white")
        metrics = ['test AUROC', 'test AUPRC']
        models = np.unique(self.results['Model Type'].values).tolist()
        models = ['linear', 'rf', 'svm', 'mlp']
        self.scores['subsample_ratio'].fillna(0.0, inplace=True)
        self.scores['subsample_ratio'] = self.scores['subsample_ratio'].astype(str)
        order = ['0.05', '0.1', '0.2', '0.3']
        self.scores['subsample_ratio'] = pd.Categorical(self.scores['subsample_ratio'], categories=order, ordered=True)
        for model in models:
            model_to_plot = self.scores[(self.scores['Model Type'] == model)]
            for metric in metrics:
                plt.figure(figsize=(12, 10))

                ax = sns.barplot(data=model_to_plot,
                                 x='subsample_ratio',
                                 y=metric,
                                 hue='Augmentation Type',
                                 dodge=True,
                                 errwidth=2,
                                 palette="Blues")

                plt.title(f'{metric.replace("test ", "")} - {model} - training sample ratio of {self.dataset}', fontsize=22)
                plt.xlabel('training sample ratio', fontsize=18)
                plt.ylabel(metric, fontsize=18)
                
                plt.xticks(rotation=45, fontsize=12)
                plt.ylim(0.5, 1.05)

                self._annotate_bar_plot(ax)
                
                handles, labels = ax.get_legend_handles_labels()
                if labels:
                    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                
                def ttest_rel_less(group_data1, group_data2, **stats_params):
                    return stats.ttest_rel(group_data1, group_data2, alternative='less', **stats_params)
                
                custom_long_name = 't-test paired samples (less)'
                custom_short_name = 't-test_rel_greater'
                custom_func = ttest_rel_less
                custom_stat_test = StatTest(custom_func, custom_long_name, custom_short_name)

                pairs = [((ratio, 'None'), (ratio, 'phylomix')) for ratio in order]

                annotator = Annotator(ax, pairs, data=model_to_plot, x='subsample_ratio', y=metric, order=order, hue='Augmentation Type')
                annotator.configure(test=custom_stat_test, text_format='full', show_test_name=False)
                annotator.apply_and_annotate()

                directory = f'../plots/{self.dataset}'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filename = f'{metric}_{model}_subsample_ratio.pdf'
                full_path = os.path.join(directory, filename)
                plt.savefig(full_path)
                plt.show()

    def perform_statistical_tests(self, comparison_df, metric_name, model, metric):
        types = comparison_df['Type'].unique()
        summary_results = []
        for type in types:
            none_scores = comparison_df[comparison_df['Type'] == type].sort_values(by='Seed')['None Score'].values
            model_scores = comparison_df[comparison_df['Type'] == type].sort_values(by='Seed')[f'{metric_name} Score'].values
            if len(none_scores) == len(model_scores) and len(none_scores) > 0:
                stat, p_value = stats.ttest_rel(model_scores, none_scores, alternative='greater')
                significant = p_value < 0.05
                summary_results.append((model, metric, type, significant, stat, p_value))
        return summary_results

    def print_summary_results(self, summary_results, results_dir):
        summary_file = os.path.join(results_dir, "summary_results.txt")
        with open(summary_file, "w") as file:
            file.write("Overall Statistical Test Results\n" + "="*50 + "\n")
            for model, metric, model_type, significant, stat, p_value in summary_results:
                stat = float(stat) if isinstance(stat, str) else stat
                p_value = float(p_value) if isinstance(p_value, str) else p_value

                line = f"Model: {model}, Metric: {metric}, Compared with: {model_type} - {'Significant' if significant else 'Not Significant'} (stat: {stat:.2f}, p-value: {p_value:.2e})\n"
                file.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='The dataset to analyze')
    parser.add_argument('--target', type=str, required=True, help='The target to analyze')
    parser.add_argument('--visualize', type=str, required=True, choices=['scores', 'time', 'curves', 'significance', 'subsample'], help='The type of visualization to create')
    parser.add_argument('--contrastive_learning', action='store_true', help='Whether to analyze contrastive learning results')
    args = parser.parse_args()

    analyzer = ResultsAnalyzer('../output/')
    analyzer._load_results(args.dataset, args.target, False, args.contrastive_learning)
    analyzer._compute_metrics(contrastive_learning=args.contrastive_learning)
    if args.visualize == 'scores':
        analyzer._visualize_test_scores(deeplearning=True, contrastive_learning=args.contrastive_learning)
        analyzer._visualize_test_scores(deeplearning=False, contrastive_learning=args.contrastive_learning)
    elif args.visualize == 'time':
        analyzer._visualize_time_elapsed()
    elif args.visualize == 'curves':
        analyzer._visualize_val_curves()
    elif args.visualize == 'significance':
        analyzer._visualize_test_scores_dot_plot(contrastive_learning=args.contrastive_learning)
    elif args.visualize == 'subsample':
        analyzer._visualize_subsample()
