# 10/15 專題企劃準備稿

## 研究問題與範圍
題目：以網站特徵與 L2 正則化邏輯斯迴歸辨識釣魚網站。
目的：利用網站特徵辨識釣魚網站，降低使用者受騙、帳號密碼及個人資訊外洩的風險。「資訊轟炸」屬於資訊過載或垃圾訊息問題，本資料沒有訊息頻率或使用者行為，無法直接評估此效果。若希望降低警告轟炸，可把誤報率列為評估指標。

## 資料來源與核對
UCI Phishing Websites：https://archive.ics.uci.edu/dataset/327/phishing+websites
資料集 DOI：https://doi.org/10.24432/C51W2X
UCI 說明主要收集來源為 PhishTank、MillerSmiles 與 Google 搜尋操作，2015 年捐贈。使用者已確認參考資料網站為 UCI 官方資料集頁面，本專題以 UCI 作為正式資料來源。
本地 Training Dataset.arff 和 phishing_dataset.csv 的 30 個特徵逐列一致，CSV 僅將 Result=-1 改為 0，Result=1 保持 1。
11,055 筆、30 個輸入特徵、1 個 Result 欄位；未發現空值或 ?。沒有原始 URL、網域 ID 或時間欄位，不能做真正的按網域或時間外部泛化驗證。

| 類別 | ARFF | CSV | 筆數 | 比例 |
|---|---|---|---|---|
| 釣魚 | -1 | 0 | 4,898 | 44.31% |
| 合法 | 1 | 1 | 6,157 | 55.69% |

一般特徵使用 -1、0、1 等規則编码，但不同欄位具有不同值域，不能一律把所有 0 當作缺失值或疑似網站。Redirect 僅有 0/1。這些是已萃取及離散化特徵，URL_Length 不是原始字元長度，web_traffic 也不是原始訪客數。特徵規則是歷史資料定義，不應宣稱為今日所有網站必然成立的安全規則。

## 30 個特徵
| # | CSV 欄位 | 意義 | 本檔值域 |
|---|---|---|---|
| 1 | having_IP_Address | 網址是否使用 IP 位址 | -1, 1 |
| 2 | URL_Length | 網址長度所屬區間 | -1, 0, 1 |
| 3 | Shortining_Service | 是否使用縮網址服務 | -1, 1 |
| 4 | having_At_Symbol | 網址是否含 @ | -1, 1 |
| 5 | double_slash_redirecting | 網址中 // 的位置是否異常 | -1, 1 |
| 6 | Prefix_Suffix | 網域是否含連字號前綴或後綴 | -1, 1 |
| 7 | having_Sub_Domain | 子網域數量所屬區間 | -1, 0, 1 |
| 8 | SSLfinal_State | HTTPS 與憑證狀態 | -1, 0, 1 |
| 9 | Domain_registeration_length | 網域註冊剩餘期限 | -1, 1 |
| 10 | Favicon | 網站圖示是否來自外部網域 | -1, 1 |
| 11 | port | 連接埠狀態是否符合規則 | -1, 1 |
| 12 | HTTPS_token | 網域名稱是否包含 HTTPS 字樣 | -1, 1 |
| 13 | Request_URL | 網頁資源來自外部網域的比例 | -1, 1 |
| 14 | URL_of_Anchor | 超連結的外部或異常比例 | -1, 0, 1 |
| 15 | Links_in_tags | Meta、Script、Link 標籤中的外部連結比例 | -1, 0, 1 |
| 16 | SFH | 表單提交目的地是否為空白或其他網域 | -1, 0, 1 |
| 17 | Submitting_to_email | 是否以 mail 或 mailto 提交資訊 | -1, 1 |
| 18 | Abnormal_URL | 網址與 WHOIS 身分是否一致 | -1, 1 |
| 19 | Redirect | 重新導向次數所屬區間 | 0, 1 |
| 20 | on_mouseover | 滑鼠移入是否改寫狀態列 | -1, 1 |
| 21 | RightClick | 是否停用滑鼠右鍵 | -1, 1 |
| 22 | popUpWidnow | 彈出視窗是否包含輸入欄位 | -1, 1 |
| 23 | Iframe | 是否使用 iframe 重新導向 | -1, 1 |
| 24 | age_of_domain | 網域年齡是否達門檻 | -1, 1 |
| 25 | DNSRecord | 是否存在 DNS 紀錄 | -1, 1 |
| 26 | web_traffic | 網站流量排名所屬區間 | -1, 0, 1 |
| 27 | Page_Rank | PageRank 是否達門檻 | -1, 1 |
| 28 | Google_Index | 是否被 Google 收錄 | -1, 1 |
| 29 | Links_pointing_to_page | 指向網站的外部連結數量所屬區間 | -1, 0, 1 |
| 30 | Statistical_report | 是否出現在歷史釣魚網域或 IP 統計名單 | -1, 1 |

## 重複與衝突資料
完全相同的 31 欄資料有 5,206 筆額外重複列；保留唯一列後為 5,849 筆（釣魚 3,019、合法 2,830，約 51.62% / 48.38%）。只有 30 個特徵的唯一組合有 5,785 組，其中 64 組同時有兩種標籤。
推薦主分析保留原始列及其頻率，以 30 個特徵的組合建立 group ID，按組切分，使相同 X 不跨訓練、驗證、測試。這些只是相同特徵組合，不代表能確認是相同網站。衝突標籤先保留，不擅自更正。另做唯一列版本的敏感度分析，報告與主分析的差異。不可讓 X 或 group ID 包含 Result。

## 模型與參數設計
主模型是 binary logistic regression（邏輯斯迴歸分類），不是以平方誤差訓練的線性回歸。
z = w^T x + b；p = sigmoid(z) = 1/(1+exp(-z))。
先保留全部 30 個特徵：30 個可學習權重，加 1 個截距，共 31 個可學習參數；不要加入 Result 作輸入。初版不需要隱藏層或先做特徵選擇。數值直接輸入等於假設離散值在 log-odds 上具有線性效果，未來可用 one-hot 編碼作比較，但維度及權重數將改變。
建議不改原 CSV，在程式內建立 y_phishing=1-Result，令 p 表示釣魚機率；p>=0.5 判釣魚。若維持原 Result，p 表示合法機率，釣魚 Recall 必須設定 pos_label=0。

目標函數：J = -(1/n) Σ[y log(p)+(1-y)log(1-p)] + (λ/2)||w||²。
建議不懲罰截距 b。這裡 λ 是 L2 強度，α 是梯度下降學習率，σ(z) 是 sigmoid 函數，不是要調的超參數。如果你原本的 sigma 指 L2 強度或初始化標準差，需明確改名及定義；目前數值全部待決定。另需記錄最大迭代數、停止條件和 random seed。若使用 sklearn LogisticRegression，主要是 C 控制正則化（越小越強）；它不提供同樣意義的學習率 α，不能直接把 λ 的數值當作 C。

## 切分、縮放與驗證
1. 先建立 group ID，再按組切分為約 80% 訓練集、10% 驗證集、10% 測試集；盡量維持類別比例，組限制下筆數可能不剛好。
2. 僅在 80% 訓練集內作 5-fold StratifiedGroupKFold，比較縮放方案與超參數。每個 fold 都重新 fit 前處理與模型。獨立 10% 驗證集用來確認所選方案及日後選擇 threshold；10% 測試集不參與調整。
3. 本資料已在 -1 到 1 的小範圍，min-max normalization 沒有明顯必要；不推薦逐列單位長度 Normalizer，會改變原始特徵關係。
4. 對 L2 + 梯度下降，建議以 StandardScaler 為初版方案，同時在 CV 比較不縮放版本。標準化不是把資料變成常態分布，而是減均值、除標準差；只在每折訓練部分 fit，再 transform 該折驗證部分。選定方案後，在 80% 訓練集 fit，使用同一 scaler transform 獨立驗證集與測試集。
5. CV 決定縮放方案與超參數後，在 80% 訓練集重訓，用 10% 驗證集確認表現；threshold 暫定 0.5，未來若調整只用驗證集。凍結全部設定後，最後評估 10% 測試集。初版不合併訓練與驗證集重訓，以維持流程簡單。
6. 原始類別不平衡輕微，先不做 SMOTE。baseline 使用訓練集多數類別分類器；原始全資料多數類別比例 55.69% 只作描述，不當作測試結果。可選擇加入決策樹作非線性比較。

## 評估
以釣魚為 positive：TP=釣魚判釣魚，FN=釣魚判合法，FP=合法判釣魚，TN=合法判合法。
Recall=TP/(TP+FN)，反映漏掉多少釣魚；Precision=TP/(TP+FP)，反映警告有多少可信；F1 綜合兩者；FPR=FP/(FP+TN)，反映合法網站被誤報的比例。全部報告，並附 confusion matrix、Accuracy 與 CV 平均±標準差。只看 Recall 會容許全部判釣魚得到 100% Recall，因此不可單獨使用。若目前尚無誤報容忍標準，建議 CV 先以釣魚 F1 作選擇依據，並重點討論 Recall/FPR。

## 文獻分析（候選文獻，待確認是否符合期中文獻要求）
Almujahid, Haq, & Alshehri (2024). Comparative evaluation of machine learning algorithms for phishing site detection. PeerJ Computer Science, 10, e2131. https://doi.org/10.7717/peerj-cs.2131
資料：UCI 和 Mendeley 兩個資料集。方法：比較 LR、SVM、KNN、DT、RF、XGBoost 及深度模型，含特徵篩選、SMOTE、交叉驗證與調參。結果：作者報告 CNN 整體表現較佳；這是該研究的結果，不能當成本組結果或保證。
可借鏡：比較 baseline、使用多指標、清楚交代前處理。本組差異：以容易解釋的 L2 logistic regression 為主，重點檢查重複特徵組合的資料洩漏與漏報/誤報權衡。文中 SMOTE 描述涉及重採樣測試資料，值得批判性討論；本組測試集不重採樣、不用來調參。不同切分與前處理的成績不能直接公平排名。
UCI 頁面是資料集來源，並非研究論文；文獻報告仍需使用研究論文，且候選文獻尚待確認是否符合期中要求。

## Drive 準備與剩餘工作
準備 upload 資料夾：摘要、資料說明、原始 ARFF、CSV、參考文獻、AI 揭露、程式碼與 requirements/README。CSV 轉換程式待隊友提供；模型尚未訓練，無本組成效數字。組員姓名、分工、期中參考文獻版型、老師/助教 Drive 上傳位置待補。尚未上傳 Drive，沒有可確認的目標位置與分享對象；不能聲稱已完成雲端提交。

## 參考
- UCI：https://archive.ics.uci.edu/dataset/327/phishing+websites
- 特徵定義：使用者提供 Phishing Websites Features.docx.pdf
- 文獻：https://pmc.ncbi.nlm.nih.gov/articles/PMC11232597/
- 縮放：https://scikit-learn.org/stable/modules/preprocessing.html
- 分組交叉驗證：https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
