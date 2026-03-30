# 已訓練模型列表 (Trained Models)

此目錄下的 `.pth` 權重檔案體積較大 (每個約 180MB)，已根據 `.gitignore` 設定排除於版本控制之外。
以下紀錄目前實驗中產出的權重檔案資訊：

## 1. Full KGAT (Baseline)
- **目錄**: `models/full_kgat/`
- **權重檔案**:
  - `kgat_checkpoint_e2.pth`
  - `kgat_checkpoint_e4.pth`
  - `kgat_checkpoint_e6.pth`
  - `kgat_checkpoint_e8.pth`
  - `kgat_checkpoint_e10.pth`

## 2. KGAT w/o Attention (wo_attn)
- **目錄**: `models/wo_attn/`
- **權重檔案**:
  - `kgat_checkpoint_e2.pth`
  - `kgat_checkpoint_e4.pth`
  - `kgat_checkpoint_e6.pth`
  - `kgat_checkpoint_e8.pth`
  - `kgat_checkpoint_e10.pth`

## 3. KGAT w/o KG (wo_kg)
- **目錄**: `models/wo_kg/`
- **權重檔案**:
  - `kgat_checkpoint_e2.pth`
  - `kgat_checkpoint_e4.pth`

## 附註
- 所有訓練日誌均存放於 `output/logs/` 目錄中，並已包含在版本控制內。
- 若需恢復訓練或進行推理，請確保本地端存在對應的 `.pth` 檔案。
