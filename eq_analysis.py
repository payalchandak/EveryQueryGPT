import os, sys, ipdb
import pickle, pandas as pd, polars as pl
import wandb, torch
from EventStream.data.eval_queries import EVAL_TIMES, EVAL_CODES
from torchmetrics.functional.classification import binary_auroc
from torchmetrics.functional.regression import pearson_corrcoef, spearman_corrcoef, r2_score
from torchmetrics.functional import mean_squared_error
import matplotlib.pyplot as plt, seaborn as sns

verbose = False 
WANDB_RUN_ID = "487l51nc"
api = wandb.Api()
run = api.run(f"payal-collabs/EveryQueryGPT/{WANDB_RUN_ID}")
read_pth = f"{run.config['save_dir']}/specific_query_predictions/"
save_pth = f"{run.config['save_dir']}/plots/"
print(f'saving plots to {save_pth}')
os.makedirs(save_pth, exist_ok=True)

df = []
for filename in os.listdir(read_pth): 
    data = {'query':filename.rstrip('.pkl'),}
    data['code'], data['offset'], data['duration'] = data['query'].split('_')
    with open(read_pth+filename,'rb') as file: 
        results = pickle.load(file)
    y_prob = torch.cat(results['zero_prob'])
    y_true = torch.cat(results['zero_truth'])
    if  (len(y_true) == sum(y_true)): 
        if verbose: print(f"Skipping {data['query']}, query event never occurs in dataset")
        continue 
    data['auroc'] = binary_auroc(y_prob, y_true.long()).item()
    if len([x for x in results['truncated_answer'] if x.nelement()]): 
        t_rate = torch.hstack([x for x in results['truncated_rate'] if x.nelement()])
        t_ans = torch.hstack([x for x in results['truncated_answer'] if x.nelement()])
        # data['truncated_r2_score'] = r2_score(t_rate, t_ans).item()
        data['truncated_mse'] = mean_squared_error(t_rate, t_ans).item()
        data['truncated_pearson'] = pearson_corrcoef(t_rate, t_ans.float()).item()
        data['truncated_spearman'] = spearman_corrcoef(t_rate, t_ans.float()).item()
    # rate = torch.cat(results['rate'])
    # ans = torch.cat(results['answer'])
    # if min(rate)!=max(rate): # output is constant when all zeros 
    #     data['overall_r2_score'] = r2_score(rate, ans)
    df.append(data)
    if verbose: print(f"{data['query']}\n{data['auroc'], data['truncated_corrcoef']}\n")

df = pd.DataFrame(df)
df['duration'] = df['duration'].astype(int)
df['offset'] = df['offset'].astype(int)
ignore = ['Endometriosis','Androgenic alopecia','Senile dementia','Pregnancy','Prostate cancer','Senile osteoporosis']
df = df.query('code not in @ignore')
df = df.query('offset==0')

for duration in df.get('duration').drop_duplicates().values: 
    df_filtered = df.query('duration==@duration').get(['code','auroc','truncated_pearson','truncated_spearman'])
    df_sorted = df_filtered.sort_values(by='auroc', ascending=False)
    df_melted = pd.melt(df_sorted, id_vars='code', value_vars=['auroc','truncated_pearson','truncated_spearman'],
                        var_name='Metric', value_name='Value')
    metric_mapping = {
        'auroc': 'AUROC of Zero Predictions', 
        'truncated_pearson': 'Pearson Corrcoef of Truncated Rate Predictions',
        'truncated_spearman': 'Spearman Corrcoef of Truncated Rate Predictions',
    }
    df_melted['Metric'] = df_melted['Metric'].map(metric_mapping)
    plt.figure(figsize=(14, 20))
    combined_plot = sns.barplot(x='Value', y='code', hue='Metric', data=df_melted, palette=['skyblue', 'lightcoral','lightgreen'])
    combined_plot.set_title(f"{duration} days", fontsize=16)
    combined_plot.set_xlabel('', fontsize=14)
    combined_plot.set_ylabel('') 
    combined_plot.tick_params(axis='y', labelsize=10)
    combined_plot.set_xlim(-1, 1)
    plt.tight_layout(pad=4.0)
    plt.savefig(f"tmp_plots/{duration}.png")
    plt.savefig(f"{save_pth}/{duration}.png")

plt.figure(figsize=(10, 6))
sc = plt.scatter(df['auroc'], df['truncated_pearson'], c=df['duration'], marker='x', alpha=0.7, cmap='viridis')
plt.colorbar(sc, label='Duration')
plt.xlabel('AUROC of Zero Predictions')
plt.ylabel('Pearson Corrcoef of Truncated Rate Predictions')
plt.xlim(0.5, 1)
plt.ylim(-1, 1)
plt.grid(True)
plt.savefig(f"tmp_plots/durations.png")
plt.savefig(f"{save_pth}/durations.png")

