"""檢查資料概況、重複列與衝突標籤；不修改資料。"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--data', type=Path,
        default=Path(__file__).resolve().parents[1] / 'data/phishing_dataset.csv',
        help='要檢查的 CSV 路徑',
    )
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    if 'Result' not in df.columns:
        parser.error('CSV 缺少 Result 標籤欄位')
    X = df.drop(columns='Result')
    if X.empty:
        parser.error('資料必須包含至少一筆資料與一個特徵')

    print('=== 資料概況 ===')
    print('資料筆數：', len(df))
    print('特徵數：', X.shape[1])
    print('缺失值總數：', int(df.isna().sum().sum()))

    print('\n=== 標籤分布：0＝釣魚，1＝合法 ===')
    print(df['Result'].value_counts(dropna=False).sort_index().to_string())
    invalid_labels = df['Result'].isna() | ~df['Result'].isin([0, 1])
    print('無效或缺失標籤筆數：', int(invalid_labels.sum()))
    print('含非預期特徵值的資料筆數：', int((~X.isin([-1, 0, 1])).any(axis=1).sum()))

    print('\n=== 重複資料 ===')
    print('整列重複筆數（不含首次出現）：', int(df.duplicated().sum()))
    print('去重後資料筆數：', len(df.drop_duplicates()))
    print('不同特徵組合數：', len(X.drop_duplicates()))
    label_counts = df.groupby(list(X.columns), dropna=False)['Result'].nunique()
    print('相同特徵卻有不同標籤的組合數：', int((label_counts > 1).sum()))


if __name__ == '__main__':
    main()
