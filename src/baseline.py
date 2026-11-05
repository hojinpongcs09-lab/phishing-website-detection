"""多數類別 baseline；使用主模型既有切分，預設只評估驗證集。"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, average_precision_score)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate-test', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    frame = pd.read_csv(root/'data/phishing_dataset.csv')
    X, y = frame.drop(columns='Result'), 1-frame.Result
    splits = pd.read_csv(root/'results/holdout80/splits.csv')
    rows = {name: splits.loc[splits.split==name, 'row_index'].to_numpy()
            for name in ['train','validation','test']}
    assert len(splits)==len(frame) and splits.row_index.nunique()==len(frame)
    groups = pd.factorize(pd.MultiIndex.from_frame(X), sort=True)[0]
    sets = [set(groups[rows[name]]) for name in rows]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    # 多數類別僅由訓練標籤決定，不使用驗證／測試集比例。
    model = DummyClassifier(strategy='most_frequent')
    model.fit(X.iloc[rows['train']], y.iloc[rows['train']])
    majority = int(model.classes_[np.argmax(model.class_prior_)])
    report = {'strategy':'most_frequent', 'positive_class':'phishing',
              'training_class_counts': {str(int(k)):int(v) for k,v in y.iloc[rows['train']].value_counts().items()},
              'predicted_class':majority, 'split_rows':{k:len(v) for k,v in rows.items()},
              'test_evaluated':args.evaluate_test}
    for name in ['validation'] + (['test'] if args.evaluate_test else []):
        idx=rows[name]
        pred=model.predict(X.iloc[idx])
        probability=model.predict_proba(X.iloc[idx])[:,list(model.classes_).index(1)]
        tn,fp,fn,tp=confusion_matrix(y.iloc[idx],pred,labels=[0,1]).ravel()
        report[name]={'accuracy':accuracy_score(y.iloc[idx],pred),
                      'precision':precision_score(y.iloc[idx],pred,zero_division=0),
                      'recall':recall_score(y.iloc[idx],pred,zero_division=0),
                      'f1':f1_score(y.iloc[idx],pred,zero_division=0),
                      'fpr':float(fp/(fp+tn)), 'roc_auc':roc_auc_score(y.iloc[idx],probability),
                      'average_precision':average_precision_score(y.iloc[idx],probability),
                      'confusion_matrix_labels_0_legitimate_1_phishing':[[int(tn),int(fp)],[int(fn),int(tp)]]}
    out=root/'results/baseline'; out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
