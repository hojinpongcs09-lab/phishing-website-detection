"""Group-disjoint L2 logistic regression; phishing is the positive class."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/phishing_dataset.csv'))
    parser.add_argument('--output', type=Path, default=Path('results/main'))
    parser.add_argument('--unique', action='store_true', help='Drop duplicate complete rows')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    frame = pd.read_csv(args.data)
    if frame.shape[1] != 31 or frame.isna().any().any():
        raise ValueError('Expected 30 features, Result, and no missing values')
    if 'Result' not in frame or set(frame.Result.unique()) != {0, 1}:
        raise ValueError('Expected CSV labels: 0=phishing, 1=legitimate')
    original_rows = len(frame)
    duplicate_rows = int(frame.duplicated().sum())
    if args.unique:
        frame = frame.drop_duplicates().reset_index(drop=True)
    X = frame.drop(columns='Result')
    if not X.isin([-1, 0, 1]).all().all():
        raise ValueError('Unexpected feature values')
    y = 1 - frame.Result
    # Exact tuples avoid hash-collision ambiguity. Conflicting labels stay together.
    groups = pd.factorize(pd.MultiIndex.from_frame(X), sort=True)[0]
    conflicts = int(pd.DataFrame({'group': groups, 'y': y}).groupby('group').y.nunique().gt(1).sum())
    # Two held-out folds give approximately 80/10/10 by row, not exact counts.
    outer = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=args.seed)
    folds = [test for _, test in outer.split(X, y, groups)]
    test, valid = folds[0], folds[1]
    train = np.concatenate(folds[2:])
    group_sets = [set(groups[index]) for index in (train, valid, test)]
    if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i)):
        raise RuntimeError('Group leakage detected')
    pipeline = Pipeline([('scale', 'passthrough'),
                         ('model', LogisticRegression(penalty='l2', solver='lbfgs', max_iter=3000))])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    search = GridSearchCV(pipeline, {'scale': ['passthrough', StandardScaler()],
                                    'model__C': [0.01, 0.1, 1, 10, 100]},
                          scoring='average_precision', cv=cv, n_jobs=1, error_score='raise')
    search.fit(X.iloc[train], y.iloc[train], groups=groups[train])
    model = search.best_estimator_
    # Validation-only threshold selection; ties favor lower FPR, then higher threshold.
    probability = model.predict_proba(X.iloc[valid])[:, 1]
    candidates = np.unique(np.r_[0, probability, 0.5, 1])
    def score(threshold):
        pred = probability >= threshold
        tn, fp, fn, tp = confusion_matrix(y.iloc[valid], pred, labels=[0, 1]).ravel()
        return (f1_score(y.iloc[valid], pred, zero_division=0), -fp / (fp + tn), threshold)
    threshold = float(max(candidates, key=score))
    # Frozen test used once after hyperparameter and threshold selection.
    probability = model.predict_proba(X.iloc[test])[:, 1]
    pred = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y.iloc[test], pred, labels=[0, 1]).ravel()
    report = {'seed': args.seed, 'unique_analysis': args.unique, 'original_rows': original_rows,
              'duplicate_complete_rows': duplicate_rows, 'unique_feature_groups': len(set(groups)),
              'conflicting_groups': conflicts, 'split_rows': dict(zip(['train','validation','test'], map(len,[train,valid,test]))),
              'group_overlap': 0, 'best_C': search.best_params_['model__C'],
              'scaling': str(search.best_params_['scale']), 'cv_average_precision': search.best_score_,
              'threshold': threshold, 'threshold_objective': 'validation phishing F1; ties lower FPR',
              'test': {'accuracy': accuracy_score(y.iloc[test], pred),
                       'precision': precision_score(y.iloc[test], pred, zero_division=0),
                       'recall': recall_score(y.iloc[test], pred), 'f1': f1_score(y.iloc[test], pred),
                       'fpr': float(fp / (fp + tn)), 'roc_auc': roc_auc_score(y.iloc[test], probability),
                       'average_precision': average_precision_score(y.iloc[test], probability),
                       'confusion_matrix_labels_0_legitimate_1_phishing': [[int(tn),int(fp)],[int(fn),int(tp)]]}}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    pd.DataFrame({'feature': X.columns, 'coefficient': model.named_steps['model'].coef_[0]}).to_csv(args.output / 'coefficients.csv', index=False)
    pd.DataFrame({'row_index': np.concatenate([train,valid,test]),
                  'split': ['train']*len(train)+['validation']*len(valid)+['test']*len(test)}).to_csv(args.output / 'splits.csv', index=False)
    joblib.dump({'pipeline': model, 'threshold': threshold, 'features': list(X.columns)}, args.output / 'model.joblib')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
