# ADR-004: 遷移至 PyTorch 原生 XPU 支援

## 狀態
已接受 (Accepted)

## 背景 (Context)
在之前的版本中，我們使用 `intel-extension-for-pytorch` (IPEX) 來支援 Intel Arc GPU 加速。然而，IPEX 的版本與原生 PyTorch 版本綁定嚴格，且在 Windows 環境下的安裝與相容性較為複雜。隨著 PyTorch 從 2.5 版本開始逐步將 Intel GPU (XPU) 支援併入主分支 (Upstreaming)，使用原生支持的編譯版本已成為更穩定且長期的選擇。

## 決策 (Decision)
1.  **升級依賴**: 將專案依賴鎖定為 `torch==2.9.1+xpu` (或更新的原生支援版本)，並使用 `https://download.pytorch.org/whl/xpu` 作為索引來源。
2.  **移除 IPEX**: 移除代碼中所有 `intel_extension_for_pytorch` 的相關導入與 `ipex.optimize` 的調用。
3.  **原生優化**: 改用 PyTorch 原生的 `torch.xpu.is_available()` 偵測裝置，並引入 `torch.compile(model)` 針對 XPU 進行運算加速。
4.  **環境管理**: 使用 `uv` 並於 `pyproject.toml` 中配置 `[[tool.uv.index]]` 以確保環境可重現性。

## 後續影響 (Consequences)
*   **優點**:
    *   環境依賴更精簡，不再需要維護 IPEX 插件。
    *   提升穩定性，減少因插件引發的記憶體洩漏或崩潰。
    *   支援 `torch.compile` 後端加速，優化 XPU 執行效率。
*   **缺點**:
    *   現階段 PyTorch 原生 XPU 仍處於快速迭代期 (Nightly/Experimental)，可能會有細微的 API 變動。
