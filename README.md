# 以網站特徵與 L2 正則化邏輯斯迴歸辨識釣魚網站

陽明 ML 第八組。以 UCI Phishing Websites 的 30 個特徵辨識釣魚網站，研究分組驗證、模型可解釋性與漏判／誤報取捨。

## 安裝與執行

```bash
make setup   # 建立 .venv 並安裝 requirements.txt
make check   # 資料完整性與重複、衝突標籤檢查
make train   # 80/10/10：訓練集內普通五折
make nested  # 80/10/10：訓練集內外層五折、內層三折
make errors  # 普通五折版本的驗證集誤判分析，需先 make train
make plots   # 普通五折版本的學習曲線與驗證診斷圖，需先 make train
```

在專案根目錄執行，Makefile 直接使用 .venv，不必手動啟用。建議 Python 3.11–3.14。

## 保留的兩種方法

兩種方法皆以相同群組切分約 80% 訓練（8,711 筆）、10% 驗證（1,191 筆）、10% 測試（1,153 筆）。以全部 30 個特徵建立群組，確保相同特徵不跨集；群組大小不同，比例不精確相等。CSV Result=0 為釣魚，程式轉為 y=1 的正類。

| 方法 | 程式 | 預設輸出 |
|---|---|---|
| 普通五折調參 | src/train.py | results/holdout80 |
| 訓練集內巢狀 5×3 折 | src/train_nested80.py | results/nested_main |

普通版在 80% 訓練集內以五折 AP 選 C 與是否標準化。巢狀版在同一訓練集內以外層五折評估、內層三折調參，報告外層 AP／ROC-AUC 平均及樣本標準差，再以完整訓練集三折選最終設定。兩者皆以 Pipeline 避免預處理洩漏。

兩種方法皆用獨立驗證集選釣魚 F1 門檻，同分優先低 FPR，再優先較高門檻。驗證集同時用於門檻選擇，因此其成績為開發結果。預設不計算測試預測；方法固定後可明確執行：

```bash
python src/train.py --evaluate-test --output results/final_test
# 或巢狀版：python src/train_nested80.py --evaluate-test --output results/final_nested_test
```

目前測試集在歷史探索中曾被查看，不能再宣稱為完全未接觸的資料。重跑會覆寫指定輸出資料夾的同名檔案。

## 模型如何訓練：L-BFGS

本專案以 L2 正則化邏輯斯迴歸進行分類，使用 L-BFGS 作為求解器：

```python
LogisticRegression(penalty="l2", solver="lbfgs", max_iter=3000)
```

模型有 30 個特徵權重與 1 個截距，共 31 個可學習參數。若選到標準化方案，輸入為標準化後的特徵；否則直接使用原始編碼。模型先計算線性分數，再透過 sigmoid 得到釣魚機率：

$$
z_i=b+\sum_{j=1}^{30}w_j\tilde{x}_{ij},\qquad
p_i=\frac{1}{1+e^{-z_i}}
$$

其中 $y_i=1$ 表示釣魚。以平均交叉熵表示時，訓練目標可寫為：

$$
L(w,b)=-\frac{1}{n}\sum_{i=1}^{n}\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]
+\frac{\lambda}{2}\sum_{j=1}^{30}w_j^2
$$

此式用於說明損失與懲罰的關係。套件以 `C` 控制正則化：C 越小，懲罰越強；C 越大，懲罰越弱。若使用平均損失，lambda 與 C 的精確換算還涉及樣本數及損失尺度，不能直接將 lambda 寫成 1/C。本專案未設定樣本權重，且 `lbfgs` 不懲罰截距。

### L-BFGS 的原理

L-BFGS（Limited-memory Broyden–Fletcher–Goldfarb–Shanno）是有限記憶的擬牛頓最佳化方法。它利用目前梯度，以及最近幾次參數與梯度的變化，近似反 Hessian 的作用，決定更新方向。Hessian 是由二階偏導數組成的矩陣，描述損失曲面的曲率。

相較於只沿負梯度更新的一般梯度下降，L-BFGS 會利用曲率資訊調整方向，並透過線搜尋選擇步長。它無須直接建立或儲存完整 Hessian 矩陣；歷史資訊透過雙迴圈遞迴計算更新方向。

一次迭代的概念流程：

1. 用目前權重與截距計算預測機率、損失及梯度。
2. 利用梯度與有限的歷史紀錄，求出更新方向。
3. 透過線搜尋決定步長，更新參數。
4. 記錄參數與梯度的變化，檢查是否收斂。

### 與交叉驗證及正則化的分工

| 設定或方法 | 作用 |
|---|---|
| L2 懲罰 | 定義對過大權重的懲罰形式 |
| `C` | 控制懲罰強弱，由訓練集內交叉驗證選擇 |
| `solver="lbfgs"` | 在給定設定下，尋找權重與截距 |
| `max_iter=3000` | 每次訓練的迭代上限，不是固定執行次數 |
| `tol` | 收斂容許誤差設定；目前使用套件預設值 |

每組候選設定、每一折都會獨立訓練縮放器與模型。選出設定後，`GridSearchCV` 自動用完整訓練集重新訓練；驗證集只選分類門檻，測試集評估固定的方法。門檻不屬於這 31 個模型參數。

若到達迭代上限仍未收斂，套件可能顯示 `ConvergenceWarning`。訓練後可從 `model.named_steps["model"].n_iter_` 查看實際迭代次數（此處 `model` 指 `train.py` 中的最佳 Pipeline）。

參考：[scikit-learn LogisticRegression 文件](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html)。期中報告原未明列 L-BFGS；此處補充目前程式實作所用的求解方法。

## 資料、產物與限制

資料來源與標籤轉換見 data/README.md；UCI 資料依 CC BY 4.0 使用。原始特徵文件與期中報告保留於 docs，原始隨機森林 notebook 僅作歷史參考，非目前訓練入口。

metrics.json 記錄設定與成績；coefficients.csv 為 30 個權重；splits.csv 記錄切分索引；model.joblib 包含縮放器、模型與門檻。巢狀版另外輸出 nested_folds.csv 與 nested_predictions.csv。

results/holdout80/plots 包含學習曲線、驗證混淆矩陣、ROC、PR、校準與門檻取捨圖。學習曲線固定已選超參數，重用驗證集，陰影為群組子抽樣標準差，非信賴區間。

特徵係數不代表因果效果；資料缺少原始 URL、網域 ID 與時間戳，無法保證網域或時間泛化。相同特徵的衝突標籤保留，不自行更正。本專案不是即時網址掃描服務。

Repo 為公開；程式與報告目前未另行授予開源授權。專案整理與程式使用 Codex 輔助，原報告的 AI 揭露保留於 PDF。

## 正式 baseline 與研究報告

Baseline 為 `DummyClassifier(strategy="most_frequent")`，僅依訓練集決定多數類別，永遠預測合法網站。L2 邏輯斯迴歸為主要模型，原始 RF 只作歷史探索紀錄。

```bash
make baseline        # 相同切分下的多數類別驗證基準，需先 make train
make regularization  # 各 C 的五折 AP 與權重大小，需先 make train
make compare         # 普通／巢狀結果比較，需先 make train、make nested
make cases           # 更新誤判分析並繪製兩筆案例
```

完整研究結果、圖表與限制見 [report.md](report.md)。普通版驗證 Accuracy 93.03%、Recall 91.59%；巢狀版驗證 Accuracy 93.12%、Recall 91.59%，兩者僅差一筆誤報。

固定普通五折模型的測試重評：Accuracy 92.45%、Recall 95.36%、FPR 9.43%，多數類別基準 Accuracy 60.71%、Recall 0%。此測試集曾於前期探索被查看，非全新獨立確認；不能與驗證成績直接排名。

`src/evaluate_fixed_models.py` 僅在明確執行時載入既有模型評估 test，不重新訓練，結果寫至 `results/final_comparison`。日常 make 指令不執行它。模型二進位不提交，重現需先 make train。
