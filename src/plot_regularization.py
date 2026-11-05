"""比較普通五折的 C 搜尋表現與完整訓練集係數大小，不使用測試資料。"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def main():
    root=Path(__file__).resolve().parents[1]
    base=root/'results/holdout80'; out=base/'plots'; out.mkdir(exist_ok=True)
    results=pd.read_csv(base/'search_results.csv')
    df=pd.read_csv(root/'data/phishing_dataset.csv')
    split=pd.read_csv(base/'splits.csv'); rows=split.loc[split.split=='train','row_index']
    X=df.drop(columns='Result').iloc[rows]; y=1-df.Result.iloc[rows]
    weights=[]
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for scale,label,color in [('passthrough','Raw features','tab:blue'),('StandardScaler()','Standardized','tab:orange')]:
        part=results.loc[results.param_scale==scale].sort_values('param_model__C')
        cs=part.param_model__C.to_numpy(dtype=float)
        axes[0].errorbar(cs,part.mean_test_score,yerr=part.std_test_score,marker='o',capsize=3,label=label,color=color)
        axes[1].plot(cs,part.mean_train_score,'o--',color=color,label=label+' train')
        axes[1].plot(cs,part.mean_test_score,'o-',color=color,label=label+' CV')
        norms=[]
        for c in cs:
            model=Pipeline([('scale',StandardScaler() if scale!='passthrough' else 'passthrough'),
                ('model',LogisticRegression(C=c,penalty='l2',solver='lbfgs',max_iter=3000))])
            model.fit(X,y)
            norm=float(np.linalg.norm(model.named_steps['model'].coef_))
            norms.append(norm); weights.append({'scale':label,'C':c,'coefficient_l2_norm':norm})
        axes[2].plot(cs,norms,'o-',label=label,color=color)
    for ax,title in zip(axes,['5-fold AP: mean +/- fold SD','Training vs CV average precision','Coefficient L2 norm (within-scale comparison)']):
        ax.set_xscale('log'); ax.set(xlabel='C (larger = weaker regularization)',title=title)
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes[0].set_ylabel('Average precision'); axes[1].set_ylabel('Average precision')
    axes[2].set_ylabel('Weight norm')
    fig.tight_layout(); fig.savefig(out/'regularization_C.png',dpi=160)
    pd.DataFrame(weights).to_csv(base/'regularization_weights.csv',index=False)
    print(results[['param_scale','param_model__C','mean_train_score','mean_test_score','std_test_score','rank_test_score']].to_string(index=False))


if __name__=='__main__':
    main()
