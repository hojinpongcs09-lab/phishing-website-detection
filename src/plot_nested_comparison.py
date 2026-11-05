"""繪製巢狀外折與相同驗證集比較，不讀取測試預測。"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).resolve().parents[1]
    ordinary=json.loads((root/'results/holdout80/metrics.json').read_text())
    nested=json.loads((root/'results/nested_main/metrics.json').read_text())
    assert ordinary['split_rows']==nested['split_rows']
    folds=nested['nested_cv']['folds']
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    for name,label,color in [('average_precision','AP','tab:blue'),('roc_auc','ROC-AUC','tab:orange')]:
        values=[row[name] for row in folds]
        mean=np.mean(values); sd=np.std(values,ddof=1)
        axes[0].plot(range(1,6),values,'o-',label=f'{label}: {mean:.5f} +/- {sd:.5f}',color=color)
        axes[0].axhline(mean,linestyle='--',alpha=.5,color=color)
    axes[0].set(xlabel='Outer fold',ylabel='Score',title='Nested 5 x 3 CV: outer evaluation',xticks=range(1,6),ylim=(.965,.985))
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)
    names=['accuracy','precision','recall','f1','fpr']; labels=['Accuracy','Precision','Recall','F1','FPR (lower better)']
    x=np.arange(len(names)); width=.36
    for offset,report,label,color in [(-width/2,ordinary,'Ordinary 5-fold','tab:blue'),(width/2,nested,'Nested 5 x 3','tab:orange')]:
        values=[report['validation'][name] for name in names]
        bars=axes[1].bar(x+offset,values,width,label=label,color=color)
        axes[1].bar_label(bars,labels=[f'{v:.4f}' for v in values],fontsize=7,padding=3)
    axes[1].set(xticks=x,xticklabels=labels,ylim=(0,1.08),ylabel='Score',title='Same validation set: thresholds selected on validation')
    axes[1].tick_params(axis='x',labelsize=8); axes[1].legend(fontsize=8)
    fig.tight_layout()
    out=root/'results/holdout80/plots'; out.mkdir(exist_ok=True)
    fig.savefig(out/'nested_comparison.png',dpi=180)
    print(out/'nested_comparison.png')


if __name__=='__main__':
    main()
