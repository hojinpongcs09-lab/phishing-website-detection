# 使用專案虛擬環境，不需要先 source activate。
PYTHON ?= python3
VENV := .venv
VENV_PYTHON := $(VENV)/bin/python
.DEFAULT_GOAL := help

.PHONY: help setup check train nested require-env

help:
	@echo "make baseline / errors / plots / regularization / compare / cases：基準與分析工具（詳 README）"
	@echo "make nested 執行 80/10/10 巢狀 5×3 折驗證"
	@echo "make setup  建立虛擬環境並安裝套件（首次使用）"
	@echo "make check  檢查資料、重複列與衝突標籤"
	@echo "make train  執行 80/10/10 主分析，只評估 valid，輸出至 results/holdout80"

$(VENV_PYTHON):
	$(PYTHON) -m venv $(VENV)

setup: $(VENV_PYTHON)
	$(VENV_PYTHON) -m pip install -r requirements.txt

# 若環境不存在，提示先安裝，避免訓練時隱式下載套件。
require-env:
	@test -x "$(VENV_PYTHON)" || (echo "請先執行 make setup"; exit 1)

check: require-env
	$(VENV_PYTHON) src/check.py

train: require-env
	$(VENV_PYTHON) src/train.py

.PHONY: errors
errors: require-env
	$(VENV_PYTHON) src/analyze_errors.py

.PHONY: plots
plots: require-env
	$(VENV_PYTHON) src/plot_diagnostics.py

.PHONY: nested
nested: require-env
	$(VENV_PYTHON) src/train_nested80.py

.PHONY: baseline
baseline: require-env
	$(VENV_PYTHON) src/baseline.py

.PHONY: regularization
regularization: require-env
	$(VENV_PYTHON) src/plot_regularization.py

.PHONY: compare
compare: require-env
	$(VENV_PYTHON) src/plot_nested_comparison.py

.PHONY: cases
cases: errors
	$(VENV_PYTHON) src/plot_error_cases.py
