"""產生訓練／驗證診斷圖；不使用測試集。"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, log_loss,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
    precision_score, recall_score, f1_score)
from sklearn.calibration import CalibrationDisplay


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / 'results/holdout80'
    out = base / 'plots'
    out.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(base / 'model.joblib')
    model, threshold = bundle['pipeline'], bundle['threshold']
    df = pd.read_csv(root / 'data/phishing_dataset.csv')
    X, y = df[bundle['features']], 1-df.Result
    splits = pd.read_csv(base / 'splits.csv')
    train = splits.loc[splits.split=='train', 'row_index'].to_numpy()
    valid = splits.loc[splits.split=='validation', 'row_index'].to_numpy()
    groups = pd.factorize(pd.MultiIndex.from_frame(X), sort=True)[0]
    assert not set(groups[train]) & set(groups[valid])
    # 每次取完整群組，不能把同特徵樣本拆開。三個種子描述子樣本波動。
    records = []
    for seed in [42, 43, 44]:
        order = np.random.default_rng(seed).permutation(np.unique(groups[train]))
        for fraction in [0.1, 0.25, 0.5, 0.75, 1.0]:
            chosen = order[:max(1, int(len(order)*fraction))]
            rows = train[np.isin(groups[train], chosen)]
            fitted = clone(model).fit(X.iloc[rows], y.iloc[rows])
            for name, idx in [('train', rows), ('validation', valid)]:
                p = fitted.predict_proba(X.iloc[idx])[:, 1]
                records.append({'seed':seed, 'fraction':fraction, 'rows':len(rows),
                    'split':name, 'average_precision':average_precision_score(y.iloc[idx],p),
                    'log_loss':log_loss(y.iloc[idx],p),
                    'accuracy_at_0.5':accuracy_score(y.iloc[idx],p>=0.5)})
    table = pd.DataFrame(records)
    table.to_csv(out/'learning_curve.csv', index=False)
    fig, axes = plt.subplots(1,3,figsize=(16,4.5))
    for ax, metric, title in zip(axes,
        ['average_precision','log_loss','accuracy_at_0.5'],
        ['Average precision (higher better)','Log loss (lower better)','Accuracy at fixed threshold 0.5']):
        for name, color in [('train','tab:blue'),('validation','tab:orange')]:
            stats=table.loc[table.split==name].groupby('fraction')[['rows',metric]].agg(['mean','std'])
            xx=stats['rows']['mean'].to_numpy(); mean=stats[metric]['mean'].to_numpy()
            sd=stats[metric]['std'].fillna(0).to_numpy()
            ax.plot(xx,mean,'o-',label=name,color=color)
            ax.fill_between(xx,mean-sd,mean+sd,alpha=.15,color=color)
        ax.set(xlabel='Training rows (whole groups)', title=title)
        ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Learning curves: fixed hyperparameters; validation reused; bands = subsample SD')
    fig.tight_layout(); fig.savefig(out/'learning_curves.png',dpi=160); plt.close(fig)
    p = model.predict_proba(X.iloc[valid])[:,1]
    fig, axes = plt.subplots(2,2,figsize=(12,10))
    ConfusionMatrixDisplay.from_predictions(y.iloc[valid],p>=threshold,
        display_labels=['Legitimate','Phishing'],ax=axes[0,0],colorbar=False)
    axes[0,0].set_title(f'Validation confusion matrix: threshold={threshold:.3f}')
    RocCurveDisplay.from_predictions(y.iloc[valid],p,ax=axes[0,1])
    PrecisionRecallDisplay.from_predictions(y.iloc[valid],p,ax=axes[1,0])
    CalibrationDisplay.from_predictions(y.iloc[valid],p,n_bins=8,strategy='quantile',ax=axes[1,1])
    axes[1,1].set_title('Validation calibration (8 quantile bins)')
    fig.tight_layout(); fig.savefig(out/'validation_diagnostics.png',dpi=160); plt.close(fig)
    ts=np.linspace(0.01,.99,99)
    fig,ax=plt.subplots(figsize=(9,5))
    for name,func in [('Precision',precision_score),('Recall',recall_score),('F1',f1_score)]:
        ax.plot(ts,[func(y.iloc[valid],p>=t,zero_division=0) for t in ts],label=name)
    ax.axvline(threshold,color='gray',linestyle='--',label='Selected validation threshold')
    ax.set(xlabel='Phishing threshold',ylabel='Score',title='Validation threshold trade-off',ylim=(0,1.02))
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(out/'threshold_tradeoff.png',dpi=160); plt.close(fig)
    print(table.loc[table.fraction==1].groupby('split')[['average_precision','log_loss','accuracy_at_0.5']].mean().to_string())
    print('Plots saved:',out)


if __name__=='__main__':
    main()
