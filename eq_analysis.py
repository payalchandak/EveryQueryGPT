import os, sys, ipdb
import pickle, pandas as pd, polars as pl
import wandb, torch
from EventStream.data.eval_queries import EVAL_TIMES, EVAL_CODES
from torchmetrics.functional.classification import binary_auroc
from torchmetrics.functional.regression import pearson_corrcoef, spearman_corrcoef, r2_score
from torchmetrics.functional import mean_squared_error
import matplotlib.pyplot as plt, seaborn as sns

class MetricsAnalysis:
    def __init__(self, wandb_run_id, verbose=False, save_to_tmp=True):
        self.verbose = verbose
        self.api = wandb.Api()
        self.run = self.api.run(f"payal-collabs/EveryQueryGPT/{wandb_run_id}")
        self.read_path = f"{self.run.config['save_dir']}/specific_query_predictions/"
        self.save_path = f"{self.run.config['save_dir']}/plots/"
        os.makedirs(self.save_path, exist_ok=True)
        self.save_to_tmp = save_to_tmp
        if self.save_to_tmp: os.makedirs('tmp/', exist_ok=True)
        self.queries_to_ignore = ['Endometriosis','Androgenic alopecia','Senile dementia','Pregnancy','Prostate cancer','Senile osteoporosis']
        self.metrics = self.local_preds_to_metrics(verbose=self.verbose)
        self.metric_map = {
            'auroc': 'AUROC of Zero Predictions', 
            'truncated_pearson': 'Pearson Corrcoef of Truncated Rate Predictions',
            'truncated_spearman': 'Spearman Corrcoef of Truncated Rate Predictions',
        }

    def local_preds_to_metrics(self, verbose=False): 
        df = []
        for filename in os.listdir(self.read_path): 
            data = {'query':filename.rstrip('.pkl'),}
            data['code'], data['offset'], data['duration'] = data['query'].split('_')
            with open(self.read_path+filename,'rb') as file: 
                results = pickle.load(file)
                results = {k:results[k] for k in results if 'loss' not in k} 
            if torch.sum(torch.cat(results['answer']))==0: 
                if verbose: print(f"Skipping {data['query']}, query event never occurs in dataset")
                continue
            results = {k:torch.hstack(results[k]) for k in results} 
            data['auroc'] = binary_auroc(results['zero_prob'], results['zero_truth'].long()).item()
            data['truncated_mse'] = mean_squared_error(results['truncated_rate'], results['truncated_answer']).item()
            data['truncated_pearson'] = pearson_corrcoef(results['truncated_rate'], results['truncated_answer'].float()).item()
            data['truncated_spearman'] = spearman_corrcoef(results['truncated_rate'], results['truncated_answer'].float()).item()
            df.append(data)
        df = pd.DataFrame(df)
        df['duration'] = df['duration'].astype(int)
        df['offset'] = df['offset'].astype(int)
        df = df.query('code not in @self.queries_to_ignore')
        return df 
    
    def plot_metrics_at_each_time(self, sort='auroc'): 
        dir = 'at_each_eval_time'
        os.makedirs(f"{self.save_path}/{dir}/", exist_ok=True)
        if self.save_to_tmp: os.makedirs(f"tmp/{dir}/", exist_ok=True)
        cols = ['code',] + list(self.metric_map.keys())
        for (offset, duration), df in self.metrics.groupby(['offset','duration']): 
            df = df.get(cols).sort_values(by=sort, ascending=False)
            df = pd.melt(df, id_vars='code', value_vars=self.metric_map.keys(), var_name='Metric', value_name='Value')
            for metric, x in df.groupby('Metric'): 
                x = x.sort_values('Value')
                plt.figure(figsize=(14, 20))
                sns.barplot(x='Value', y='code', data=x)
                name = f"Offset {offset}, Duration {duration}"
                plt.title(name, fontsize=16)
                plt.xlabel(self.metric_map[metric])
                plt.ylabel('') 
                plt.tick_params(axis='y', labelsize=10)
                if metric=='auroc': plt.xlim(0, 1)
                else: plt.xlim(-1, 1)
                plt.tight_layout(pad=4.0)
                if self.save_to_tmp: plt.savefig(f"tmp/{dir}/{metric} {name}.png")
                plt.savefig(f"{self.save_path}/{dir}/{metric} {name}.png")
                plt.close()
            df['Metric'] = df['Metric'].map(self.metric_map)
            plt.figure(figsize=(14, 20))
            combined_plot = sns.barplot(x='Value', y='code', hue='Metric', data=df, palette=['skyblue', 'lightcoral','lightgreen'])
            name = f"Offset {offset}, Duration {duration}"
            combined_plot.set_title(name, fontsize=16)
            combined_plot.set_xlabel('')
            combined_plot.set_ylabel('') 
            combined_plot.tick_params(axis='y', labelsize=10)
            combined_plot.set_xlim(-1, 1)
            plt.tight_layout(pad=4.0)
            if self.save_to_tmp: plt.savefig(f"tmp/{dir}/combined {name}.png")
            plt.savefig(f"{self.save_path}/{dir}/combined {name}.png")
            plt.close()

    def plot_metric_v_metric(
            self, 
            keep_const, 
            x_metric='auroc', 
            y_metric='truncated_pearson', 
            individual_codes=True
        ): 
        if keep_const=='offset': 
            df = self.metrics.query('offset==0')
            vary = 'duration'
        elif keep_const=='duration': 
            df = self.metrics.query('duration==30')
            vary = 'offset'
        else: raise ValueError('invalid keep_const, pick offset or duration')
        dir = f'vary_{vary}/{x_metric}_v_{y_metric}'
        os.makedirs(f"{self.save_path}/{dir}/", exist_ok=True)
        if self.save_to_tmp: os.makedirs(f"tmp/{dir}/", exist_ok=True)
        plt.figure(figsize=(10, 6))
        sc = plt.scatter(df[x_metric], df[y_metric], c=df[vary], marker='x', alpha=0.5, cmap='cool')
        plt.colorbar(sc, label=vary)
        plt.xlabel(self.metric_map[x_metric])
        plt.ylabel(self.metric_map[y_metric])
        if x_metric=="auroc": plt.xlim(0, 1)
        else: plt.xlim(-1, 1)
        if y_metric=="auroc": plt.ylim(0, 1)
        else: plt.ylim(-1, 1)
        plt.grid(True)
        if self.save_to_tmp: plt.savefig(f"tmp/{dir}/all_codes.png")
        plt.savefig(f"{self.save_path}/{dir}/all_codes.png")
        plt.close()
        if individual_codes: 
            dir = dir+'/individual_codes'
            os.makedirs(f"{self.save_path}/{dir}/", exist_ok=True)
            if self.save_to_tmp: os.makedirs(f"tmp/{dir}/", exist_ok=True)
            for code, df_code in df.groupby('code'): 
                plt.figure(figsize=(10, 6))
                sc = plt.scatter(df_code[x_metric], df_code[y_metric], c=df_code[vary], marker='x', cmap='cool')
                plt.colorbar(sc, label=vary)
                plt.title(code)
                plt.xlabel(self.metric_map[x_metric])
                plt.ylabel(self.metric_map[y_metric])
                plt.grid(True)
                if self.save_to_tmp: plt.savefig(f"tmp/{dir}/{code}.png")
                plt.savefig(f"{self.save_path}/{dir}/{code}.png")
                plt.close()

    def boxplot_metric_variation(self):
        for keep_const in ['offset', 'duration']: 
            if keep_const=='offset': 
                df = self.metrics.query('offset==0')
                vary = 'duration'
            elif keep_const=='duration': 
                df = self.metrics.query('duration==30')
                vary = 'offset'
            dir = f'vary_{vary}/metric_variation/'
            os.makedirs(f"{self.save_path}/{dir}/", exist_ok=True)
            if self.save_to_tmp: os.makedirs(f"tmp/{dir}/", exist_ok=True)
            for metric, metric_name in self.metric_map.items(): 
                plt.figure(figsize=(10,10))
                sns.boxplot(data=df, x=metric, y='code')
                plt.title(f'{metric_name} across different {vary}s')
                plt.tick_params(axis='y', labelsize=10)
                plt.tight_layout(pad=4.0)
                plt.grid(True)
                if self.save_to_tmp: plt.savefig(f"tmp/{dir}/{metric}.png")
                plt.savefig(f"{self.save_path}/{dir}/{metric}.png")
                plt.close()

m = MetricsAnalysis(wandb_run_id="487l51nc")
m.plot_metrics_at_each_time()
m.boxplot_metric_variation()
m.plot_metric_v_metric(
    keep_const='offset', 
    x_metric='auroc', 
    y_metric='truncated_pearson', 
    individual_codes=True
)
m.plot_metric_v_metric(
    keep_const='duration', 
    x_metric='auroc', 
    y_metric='truncated_pearson', 
    individual_codes=True
)