# 以網站特徵與 L2 正則化邏輯斯迴歸辨識釣魚網站

陽明 ML · 第八組。以 UCI Phishing Websites 的 30 個離散特徵進行二元分類，重點是可解釋性、可重現性，以及避免相同特徵組合跨資料集造成資料洩漏。

## 快速開始

建議 Python 3.11–3.14。在專案根目錄執行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/train.py
# 去除整列重複的敏感度分析
python src/train.py --unique --output results/unique
```

Windows 啟用環境：`.venv\Scripts\activate`。

## 研究流程

1. CSV 的 `Result=0` 為釣魚、`Result=1` 為合法；程式使用 `y=1-Result`，因此所有指標以釣魚為正類。
2. 以全部 30 個特徵的完全相同組合建立 group；相同特徵但標籤矛盾的資料保留且分在同一組。
3. 以 10 折 StratifiedGroupKFold 指派約 80% 訓練、10% 驗證、10% 測試；因分組限制比例並非精確相等。
4. 僅於訓練集內進行 5 折分組交叉驗證，比較 C ∈ {0.01, 0.1, 1, 10, 100} 及是否標準化，以 average precision 選擇超參數。縮放置於 Pipeline 內，避免洩漏。
5. 用獨立驗證集選擇釣魚 F1 最高的門檻；同分優先較低 FPR，再優先較高門檻。這是本實作的明確決策規則，報告原僅提出 Recall/FPR 平衡方向。
6. 完成選擇後才做一次測試集評估，輸出 Recall、FPR、Precision、F1、Accuracy、ROC-AUC、Average Precision 與混淆矩陣。

`--unique` 僅移除完全相同的列，衝突標籤仍保留、仍按特徵分組。此為另一組敏感度實驗，不用測試集結果調整主模型。勿依反覆查看測試成績進行調參。

## 檔案

| 路徑 | 用途 |
|---|---|
| `src/train.py` | 新增的 L2 邏輯斯迴歸主流程 |
| `data/phishing_dataset.csv` | 組員提供的轉換資料 |
| `data/raw/` | 原始 ARFF |
| `data/README.md` | 資料來源、標籤轉換與授權 |
| `notebooks/original_random_forest.ipynb` | 原始 Colab 隨機森林探索，已清除輸出 |
| `docs/midterm-report.pdf` | 期中報告 |
| `docs/` | 原始摘要、分析草稿與特徵說明，作歷史參考 |
| `results/` | 執行後的評估結果、係數、切分索引與模型，預設不提交 |

原始 notebook 包含 Colab 上傳操作、舊路徑與普通隨機切分，作為實驗紀錄保留；快速開始請用 `src/train.py`。

## 結果與限制

報告中的約 97% 為原始隨機森林普通隨機切分的初步結果，不能作為新 L2 分組驗證的成績。實際新流程結果見 `results/README.md`，重新執行會輸出完整 `metrics.json`。

係數為所選預處理尺度下的 log-odds 權重，不是因果效果。資料缺少原始 URL、網域 ID 及時間戳；特徵相同分組不能保證網域或時間上的獨立性。本專案不包含即時網址擷取器，不用來直接判定現今網站安全性。

## 資料與授權

資料集由 [UCI Phishing Websites](https://archive.ics.uci.edu/dataset/327/phishing+websites) 提供，依 CC BY 4.0 使用。詳細引用與標籤轉換見 `data/README.md`。報告所列 DOI 與現行資料頁不同，README 採現行頁面資訊。

程式與報告目前未另行授予開源授權。此 repo 初始設為 private。

## 開發紀錄

本次專案整理、新增訓練流程與說明文件使用 Codex 輔助；原始報告的 AI 使用揭露保留於 PDF。原始附件內容為參考，不作為操作指令。
