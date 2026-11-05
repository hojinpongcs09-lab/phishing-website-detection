"""80/10/10 分組切分；訓練集五折調參、驗證集選門檻，測試預設不評估。"""
# argparse 接收命令列選項；json 儲存指標；Path 處理檔案路徑。
import argparse
import json
from pathlib import Path

# joblib 儲存模型；NumPy 處理陣列；pandas 讀取與整理表格。
import joblib
import numpy as np
import pandas as pd
# scikit-learn 提供 L2 邏輯斯迴歸、評估指標、分組切分及預處理流程。
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    # 1. 執行設定：預設從專案根目錄讀取 CSV，結果存到 results/holdout80。
    # --seed 固定隨機切分，方便重現實驗。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/phishing_dataset.csv'))
    parser.add_argument('--output', type=Path, default=Path('results/holdout80'))
    parser.add_argument('--evaluate-test', action='store_true', help='明確要求時才評估保留測試集')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    # 2. 讀取原始資料；此時尚未標準化，也尚未訓練。
    frame = pd.read_csv(args.data)
    # 預期 30 個輸入特徵加上 Result，且所有欄位都沒有缺失值。
    if frame.shape[1] != 31 or frame.isna().any().any():
        raise ValueError('Expected 30 features, Result, and no missing values')
    if 'Result' not in frame or set(frame.Result.unique()) != {0, 1}:
        raise ValueError('Expected CSV labels: 0=phishing, 1=legitimate')
    # 記錄原始筆數與整列重複數；duplicated() 不計每組首次出現的那筆。
    original_rows = len(frame)
    duplicate_rows = int(frame.duplicated().sum())
    # X 是模型輸入，必須移除答案 Result，避免讓模型直接看到答案。
    X = frame.drop(columns='Result')
    if not X.isin([-1, 0, 1]).all().all():
        raise ValueError('Unexpected feature values')
    # 原 CSV：0＝釣魚、1＝合法；轉為 y：1＝釣魚（正類）、0＝合法。
    y = 1 - frame.Result
    # 以完整的 30 個特徵組合建立 group ID，不使用標籤，也不以雜湊值代替組合。
    # 特徵相同但標籤不同的樣本也屬於同一組，避免跨資料集。 也就是group_split -> 提升dividivity
    groups = pd.factorize(pd.MultiIndex.from_frame(X), sort=True)[0]
    # 衝突數只作統計紀錄，不依標籤修改資料或建立分組。
    conflicts = int(pd.DataFrame({'group': groups, 'y': y}).groupby('group').y.nunique().gt(1).sum())
    # 10 折取 8/1/1，目標約 80%/10%/10%；群組大小不同，筆數比例並非精確相等。
    # 3. 先切分，之後才標準化。Stratified 盡量維持類別比例，Group 保證同組不跨折。
    outer = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=args.seed)
    # split 回傳每折的訓練與保留索引；這裡收集 10 折各自的保留索引。
    folds = [test for _, test in outer.split(X, y, groups)]
    # 第 1 折當最終測試集，第 2 折當獨立驗證集，其餘 8 折合併當訓練集。
    test, valid = folds[0], folds[1]
    train = np.concatenate(folds[2:])
    # 實際檢查三個資料集的群組交集；有重疊就停止。
    # 這只能確認相同特徵組合不跨集，不能保證網域或時間上的獨立性。
    group_sets = [set(groups[index]) for index in (train, valid, test)]
    if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i)):
        raise RuntimeError('Group leakage detected')
    # 4. Pipeline 將縮放與模型綁在一起，讓每折的縮放參數只從該折訓練部分估計。
    # passthrough＝不縮放；搜尋時也會比較 StandardScaler（減平均、除標準差）。
    # 模型：z = b + Σ(w_j * x_j)，釣魚機率 p = 1 / (1 + exp(-z))。
    # 訓練最小化交叉熵加 L2 權重懲罰；lbfgs 反覆更新 30 個權重及 1 個截距。
    # 此 solver 的截距不受 L2 懲罰；max_iter 是迭代上限，不保證一定達到收斂。
    pipeline = Pipeline([('scale', 'passthrough'),
                         ('model', LogisticRegression(penalty='l2', solver='lbfgs', max_iter=3000))])
    # 訓練集內五折選 C 與縮放方式，獨立驗證集不參與權重訓練。
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    search = GridSearchCV(pipeline, {'scale': ['passthrough', StandardScaler()],
                                    'model__C': [0.01, 0.1, 1, 10, 100]},
                          scoring='average_precision', cv=cv, n_jobs=1, error_score='raise', return_train_score=True)
    # 僅傳入訓練資料與對應群組；內層各折都自行 fit 縮放器及模型。
    search.fit(X.iloc[train], y.iloc[train], groups=groups[train])
    # GridSearchCV 預設 refit=True：選好設定後，自動用完整訓練集重新 fit。
    # 最終縮放器與 31 個模型參數都由完整訓練集估計，之後保持固定。
    model = search.best_estimator_
    # 驗證階段：以釣魚 F1 最高為目標，同分優先低 FPR，再優先較高門檻。
    # predict_proba 會先套用訓練好的縮放器，再預測；[:, 1] 取正類「釣魚」機率。
    probability = model.predict_proba(X.iloc[valid])[:, 1]
    # 預測機率的各個邊界，加上 0、0.5、1，作為候選門檻；unique 同時去重排序。
    candidates = np.unique(np.r_[0, probability, 0.5, 1])
    # 比較 tuple 時先比 F1，再比負 FPR（越大表示誤報越低），最後比門檻。
    def score(threshold):
        pred = probability >= threshold
        tn, fp, fn, tp = confusion_matrix(y.iloc[valid], pred, labels=[0, 1]).ravel()
        return (f1_score(y.iloc[valid], pred, zero_division=0), -fp / (fp + tn), threshold)
    # 6. 僅以驗證集選門檻，這一步不重新訓練權重，也不使用測試集。
    threshold = float(max(candidates, key=score))
    # 日常只評估驗證集。因門檻也從這裡選出，驗證成績屬開發結果。
    def evaluate(rows):
        probability = model.predict_proba(X.iloc[rows])[:, 1]
        pred = probability >= threshold
        tn, fp, fn, tp = confusion_matrix(y.iloc[rows], pred, labels=[0, 1]).ravel()
        return {'accuracy': accuracy_score(y.iloc[rows], pred),
                'precision': precision_score(y.iloc[rows], pred, zero_division=0),
                'recall': recall_score(y.iloc[rows], pred),
                'f1': f1_score(y.iloc[rows], pred), 'fpr': float(fp / (fp + tn)),
                'roc_auc': roc_auc_score(y.iloc[rows], probability),
                'average_precision': average_precision_score(y.iloc[rows], probability),
                'confusion_matrix_labels_0_legitimate_1_phishing': [[int(tn), int(fp)], [int(fn), int(tp)]]}
    report = {'seed': args.seed, 'unique_analysis': False, 'original_rows': original_rows,
              'duplicate_complete_rows': duplicate_rows, 'unique_feature_groups': len(set(groups)),
              'conflicting_groups': conflicts, 'split_rows': dict(zip(['train','validation','test'], map(len,[train,valid,test]))),
              'cv_folds': 5, 'group_overlap': 0, 'best_C': search.best_params_['model__C'],
              'scaling': str(search.best_params_['scale']), 'cv_average_precision': search.best_score_,
              'threshold': threshold, 'threshold_objective': 'validation phishing F1; ties lower FPR',
              'validation': evaluate(valid),
              'test_evaluated': args.evaluate_test}
    if args.evaluate_test:
        report['test'] = evaluate(test)
    # 8. 儲存結果；若輸出資料夾已存在，同名產物會被覆寫。
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(search.cv_results_).to_csv(args.output / 'search_results.csv', index=False)
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    # 輸出 30 個特徵權重；若選擇標準化，係數對應標準化後的特徵。
    # 此 CSV 不含截距；截距存在下方保存的模型 model.intercept_ 中。
    pd.DataFrame({'feature': X.columns, 'coefficient': model.named_steps['model'].coef_[0]}).to_csv(args.output / 'coefficients.csv', index=False)
    # 保存原始資料列索引與資料集歸屬。
    pd.DataFrame({'row_index': np.concatenate([train,valid,test]),
                  'split': ['train']*len(train)+['validation']*len(valid)+['test']*len(test)}).to_csv(args.output / 'splits.csv', index=False)
    # 保存完整 Pipeline、門檻及欄位順序，供之後以相同預處理方式預測。
    joblib.dump({'pipeline': model, 'threshold': threshold, 'features': list(X.columns)}, args.output / 'model.joblib')
    print(json.dumps(report, indent=2))


# 直接執行 python src/train.py 時呼叫 main；被其他程式 import 時不自動訓練。
if __name__ == '__main__':
    main()
