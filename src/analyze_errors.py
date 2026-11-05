"""分析 holdout80 驗證集誤判，不計算測試集預測。"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'results/holdout80'
    bundle = joblib.load(output / 'model.joblib')
    frame = pd.read_csv(root / 'data/phishing_dataset.csv')
    splits = pd.read_csv(output / 'splits.csv')
    metrics = json.loads((output / 'metrics.json').read_text())
    if metrics['unique_analysis']:
        raise ValueError('本分析目前僅適用未去重的主分析')
    features = bundle['features']
    X = frame[features]
    groups = pd.factorize(pd.MultiIndex.from_frame(X), sort=True)[0]
    labels = pd.DataFrame({'group': groups, 'label': frame.Result})
    conflicting = set(labels.groupby('group').label.nunique().loc[lambda v: v > 1].index)
    rows = splits.loc[splits.split == 'validation', 'row_index'].to_numpy()
    pipeline = bundle['pipeline']
    threshold = bundle['threshold']
    probability = pipeline.predict_proba(X.iloc[rows])[:, 1]
    truth = (1 - frame.Result.iloc[rows]).to_numpy()
    predicted = (probability >= threshold).astype(int)
    result = pd.DataFrame({'row_index': rows, 'group': groups[rows],
                           'actual_phishing': truth, 'predicted_phishing': predicted,
                           'phishing_probability': probability,
                           'distance_to_threshold': np.abs(probability - threshold),
                           'conflicting_label_group': np.isin(groups[rows], list(conflicting))})
    result['outcome'] = np.select([(truth==0)&(predicted==1), (truth==1)&(predicted==0)],
                                   ['FP', 'FN'], default='correct')
    # 縮放後的 x 乘上權重，得到對 logit 的貢獻；截距另外記錄。
    transformed = pipeline[:-1].transform(X.iloc[rows])
    model = pipeline.named_steps['model']
    contributions = np.asarray(transformed) * model.coef_[0]
    logits = contributions.sum(axis=1) + model.intercept_[0]
    assert np.allclose(logits, pipeline.decision_function(X.iloc[rows]))
    top_indices = np.argsort(-np.abs(contributions), axis=1)[:, :3]
    for rank in range(3):
        idx = top_indices[:, rank]
        result[f'top_{rank+1}_feature'] = np.asarray(features)[idx]
        result[f'top_{rank+1}_logit_contribution'] = contributions[np.arange(len(rows)), idx]
    result['intercept'] = model.intercept_[0]
    errors = result.loc[result.outcome != 'correct'].sort_values('distance_to_threshold')
    errors.to_csv(output / 'validation_errors.csv', index=False)
    result.to_csv(output / 'validation_predictions.csv', index=False)
    contribution_table = pd.DataFrame(contributions, columns=features)
    contribution_table.insert(0, 'row_index', rows)
    contribution_table.to_csv(output / 'validation_contributions.csv', index=False)
    summary = {'validation_rows': len(rows), 'errors': len(errors), 'threshold': threshold}
    for category in ['FP', 'FN']:
        part = errors.loc[errors.outcome == category]
        summary[category] = {'count': len(part),
                             'conflicting_label_rows': int(part.conflicting_label_group.sum()),
                             'probability_min': float(part.phishing_probability.min()),
                             'probability_max': float(part.phishing_probability.max()),
                             'median_distance_to_threshold': float(part.distance_to_threshold.median())}
    (output / 'error_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    assert summary['FP']['count'] == metrics['validation']['confusion_matrix_labels_0_legitimate_1_phishing'][0][1]
    assert summary['FN']['count'] == metrics['validation']['confusion_matrix_labels_0_legitimate_1_phishing'][1][0]
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
