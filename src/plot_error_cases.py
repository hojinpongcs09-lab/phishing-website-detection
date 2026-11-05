"""解釋兩筆非衝突驗證誤判；特徵貢獻為 logit 單位，非因果效果。"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).resolve().parents[1]; base=root/'results/holdout80'
    errors=pd.read_csv(base/'validation_errors.csv')
    contributions=pd.read_csv(base/'validation_contributions.csv').set_index('row_index')
    data=pd.read_csv(root/'data/phishing_dataset.csv')
    bundle=joblib.load(base/'model.joblib'); threshold=bundle['threshold']
    boundary=np.log(threshold/(1-threshold))
    selected=[]
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    for ax,kind in zip(axes,['FP','FN']):
        candidates=errors.loc[(errors.outcome==kind)&(~errors.conflicting_label_group)]
        case=candidates.sort_values('phishing_probability',ascending=(kind=='FN')).iloc[0]
        row=int(case.row_index); values=contributions.loc[row]
        top=values.abs().nlargest(8).index
        components=values.loc[top].copy()
        components.loc['Other features']=values.drop(top).sum()
        components.loc['Intercept']=float(case.intercept)
        colors=['tab:red' if v>0 else 'tab:blue' for v in components]
        ax.barh(components.index,components.values,color=colors)
        ax.invert_yaxis(); ax.axvline(0,color='black',linewidth=.7)
        ax.set(xlabel='Contribution to phishing logit (+ increases risk)',
               title=f'{kind}: row {row}, P(phishing)={case.phishing_probability:.4f}')
        ax.grid(axis='x',alpha=.2)
        logit=float(values.sum()+case.intercept)
        assert np.isclose(1/(1+np.exp(-logit)),case.phishing_probability)
        selected.append({'outcome':kind,'row_index':row,'phishing_probability':float(case.phishing_probability),
                         'threshold':threshold,'logit':logit,'decision_boundary_logit':float(boundary),
                         'intercept':float(case.intercept),
                         'features':[{'feature':name,'raw_value':int(data.loc[row,name]),
                                      'logit_contribution':float(values[name])} for name in top]})
    fig.suptitle('Non-conflicting validation errors: highest-confidence FP and FN')
    fig.tight_layout(); out=base/'plots'; out.mkdir(exist_ok=True)
    fig.savefig(out/'error_case_contributions.png',dpi=170)
    (base/'error_cases.json').write_text(json.dumps(selected,indent=2)+'\n')
    for case in selected:
        print(case['outcome'], 'row',case['row_index'],'probability',case['phishing_probability'])
        print(pd.DataFrame(case['features']).to_string(index=False))


if __name__=='__main__':
    main()
