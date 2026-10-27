/**
 * 協同知識圖注意力推薦與忠實度驗證系統 - 前端主邏輯腳本
 * 支援桌機連線推論與本機離線展示無縫切換，落實學術規範與快取防呆機制。
 */

(function () {
  "use strict";

  // 全域狀態
  const state = {
    serverUrl: "http://localhost:8501",
    isConnected: false,
    demoData: null,
    currentUserId: 17049,
    currentDepth: "3",
    currentCaseId: "deep_propagation",
    currentRecommendData: null,
    currentProbeMode: "normal",
    selectedPathIndex: 0,
    customUserResults: {}, // 自訂使用者結果快取字典 { uid: { depth: recData } }
    isComputing: false,
    activeAbortController: null,
    currentTaskLabel: "",
    backgroundTask: null, // { userId, depth, controller, result, isCompleted, label }
    isBackgroundMode: false,
  };

  // DOM 元素快取
  const dom = {
    serverConfigBtn: document.getElementById("serverConfigBtn"),
    serverModal: document.getElementById("serverModal"),
    serverModalCloseBtn: document.getElementById("serverModalCloseBtn"),
    testServerBtn: document.getElementById("testServerBtn"),
    saveServerBtn: document.getElementById("saveServerBtn"),
    serverModalFeedback: document.getElementById("serverModalFeedback"),

    serverUrlInput: document.getElementById("serverUrlInput"),
    reconnectBtn: document.getElementById("reconnectBtn"),
    connectionStatus: document.getElementById("connectionStatus"),
    taskStatusPill: document.getElementById("taskStatusPill"),
    taskStatusPillText: document.getElementById("taskStatusPillText"),
    pillSpinner: document.getElementById("pillSpinner"),

    depthButtonGroup: document.getElementById("depthButtonGroup"),
    caseSelect: document.getElementById("caseSelect"),
    userIdInput: document.getElementById("userIdInput"),
    userSelect: document.getElementById("userSelect"),
    runInferenceBtn: document.getElementById("runInferenceBtn"),

    testHitBadge: document.getElementById("testHitBadge"),
    recipeName: document.getElementById("recipeName"),
    recipeId: document.getElementById("recipeId"),
    origProbVal: document.getElementById("origProbVal"),
    origProbBaseFill: document.getElementById("origProbBaseFill"),
    origProbFill: document.getElementById("origProbFill"),

    userInteractionsVal: document.getElementById("userInteractionsVal"),
    itemDegreeVal: document.getElementById("itemDegreeVal"),
    modelHrVal: document.getElementById("modelHrVal"),
    modelNdcgVal: document.getElementById("modelNdcgVal"),

    caseInsightBox: document.getElementById("caseInsightBox"),
    caseInsightTitle: document.getElementById("caseInsightTitle"),
    caseInsightDesc: document.getElementById("caseInsightDesc"),

    fidelitySection: document.getElementById("fidelitySection"),
    noPathBadge: document.getElementById("noPathBadge"),
    probeNormalBtn: document.getElementById("probeNormalBtn"),
    probeFPlusBtn: document.getElementById("probeFPlusBtn"),
    probeFMinusBtn: document.getElementById("probeFMinusBtn"),
    cardFPlus: document.getElementById("cardFPlus"),
    fPlusVal: document.getElementById("fPlusVal"),
    cardFMinus: document.getElementById("cardFMinus"),
    fMinusVal: document.getElementById("fMinusVal"),
    barOrigProb: document.getElementById("barOrigProb"),
    barOrigText: document.getElementById("barOrigText"),
    barProbeProb: document.getElementById("barProbeProb"),
    barProbeText: document.getElementById("barProbeText"),
    probeAcademicRationale: document.getElementById("probeAcademicRationale"),
    promptNormal: document.getElementById("promptNormal"),
    promptFPlus: document.getElementById("promptFPlus"),
    promptFMinus: document.getElementById("promptFMinus"),

    graphSection: document.getElementById("graphSection"),
    svgBgDecoGroup: document.getElementById("svgBgDecoGroup"),
    svgEdgesGroup: document.getElementById("svgEdgesGroup"),
    svgNodesGroup: document.getElementById("svgNodesGroup"),
    graphEmptyMsg: document.getElementById("graphEmptyMsg"),
    graphEmptyTitle: document.getElementById("graphEmptyTitle"),
    graphEmptyDesc: document.getElementById("graphEmptyDesc"),
    pathsSelector: document.getElementById("pathsSelector"),

    inferenceModal: document.getElementById("inferenceModal"),
    modalStageTitle: document.getElementById("modalStageTitle"),
    modalStageDesc: document.getElementById("modalStageDesc"),
    modalBackgroundBtn: document.getElementById("modalBackgroundBtn"),
    modalAbortBtn: document.getElementById("modalAbortBtn"),
  };

  /**
   * 初始化系統
   */
  async function init() {
    const savedUrl = localStorage.getItem("kgat_server_url");
    if (savedUrl) {
      state.serverUrl = savedUrl;
      if (dom.serverUrlInput) dom.serverUrlInput.value = savedUrl;
    }
    bindEvents();
    await loadOfflineData();
    await checkConnection();
    executeInference();
  }

  /**
   * 載入離線評估快取
   */
  async function loadOfflineData() {
    if (window.DEMO_CASES_DATA) {
      state.demoData = window.DEMO_CASES_DATA;
      populateUserDropdown();
      return;
    }

    try {
      const res = await fetch("demo_cases.json");
      if (res.ok) {
        state.demoData = await res.json();
        populateUserDropdown();
      }
    } catch (err) {
      console.warn("離線快取載入異常，將依賴連線伺服器：", err);
    }
  }

  /**
   * 填入使用者選單
   */
  function populateUserDropdown() {
    if (!state.demoData || !state.demoData.users) return;
    const users = state.demoData.users;
    dom.userSelect.innerHTML = "";

    users.forEach((u) => {
      const opt = document.createElement("option");
      opt.value = u.user_id;
      opt.textContent = `使用者 ${u.user_id}`;
      dom.userSelect.appendChild(opt);
    });

    dom.userSelect.value = state.currentUserId;
    if (dom.userIdInput) {
      dom.userIdInput.value = state.currentUserId;
    }
  }

  /**
   * 探測後端推論伺服器連線狀態（連線狀態唯一維護點）
   */
  async function checkConnection() {
    const url = (dom.serverUrlInput ? dom.serverUrlInput.value.trim() : state.serverUrl).replace(/\/+$/, "");
    state.serverUrl = url;

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 2000);
      const res = await fetch(`${url}/api/status`, {
        method: "GET",
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (res.ok) {
        state.isConnected = true;
        setConnectionBadge(true);
        return true;
      }
    } catch (e) {
      // 連線逾時或失敗
    }

    state.isConnected = false;
    setConnectionBadge(false);
    return false;
  }

  /**
   * 更新連線狀態標籤（嚴格僅保留「● 連線」與「○ 離線」）
   */
  function setConnectionBadge(isOnline) {
    if (isOnline) {
      dom.connectionStatus.className = "status-badge online";
      dom.connectionStatus.textContent = "● 連線";
    } else {
      dom.connectionStatus.className = "status-badge offline";
      dom.connectionStatus.textContent = "○ 離線";
    }
  }

  /**
   * 顯示中央聚焦遮罩
   */
  function showInferenceModal(title, desc) {
    dom.modalStageTitle.textContent = title || "推論計算中";
    dom.modalStageDesc.textContent = desc || `正在為使用者 ${state.currentUserId} 進行模型評估...`;
    dom.inferenceModal.style.display = "flex";
  }

  /**
   * 隱藏中央聚焦遮罩
   */
  function hideInferenceModal() {
    dom.inferenceModal.style.display = "none";
  }

  /**
   * 顯示背景狀態膠囊 (支援進行中與已就緒完成雙態)
   */
  function showTaskPill(label, isCompleted = false) {
    if (label) state.currentTaskLabel = label;
    dom.taskStatusPillText.textContent = state.currentTaskLabel || `● 推論中：使用者 ${state.currentUserId}`;
    if (dom.pillSpinner) {
      dom.pillSpinner.style.display = isCompleted ? "none" : "inline-block";
    }
    if (isCompleted) {
      dom.taskStatusPill.classList.add("completed");
    } else {
      dom.taskStatusPill.classList.remove("completed");
    }
    dom.taskStatusPill.style.display = "inline-flex";
  }

  /**
   * 隱藏背景狀態膠囊
   */
  function hideTaskPill() {
    dom.taskStatusPill.style.display = "none";
    dom.taskStatusPill.classList.remove("completed");
    if (dom.pillSpinner) {
      dom.pillSpinner.style.display = "inline-block";
    }
  }

  /**
   * 中斷當前運算
   */
  function abortCurrentOperation() {
    if (state.activeAbortController) {
      state.activeAbortController.abort();
      state.activeAbortController = null;
    }
    if (state.backgroundTask && state.backgroundTask.controller) {
      state.backgroundTask.controller.abort();
      state.backgroundTask = null;
    }
    state.isComputing = false;
    state.isBackgroundMode = false;
    hideInferenceModal();
    hideTaskPill();
    dom.runInferenceBtn.classList.remove("btn-loading");
    dom.runInferenceBtn.textContent = "執行推論";
  }

  /**
   * 使用者編號邊界驗證 (0 ~ 226569)
   */
  function validateUserId(val) {
    const num = parseInt(val, 10);
    const isValid = !isNaN(num) && num >= 0 && num <= 226569;
    if (isValid) {
      dom.userIdInput.classList.remove("input-error");
      dom.runInferenceBtn.disabled = false;
    } else {
      dom.userIdInput.classList.add("input-error");
      dom.runInferenceBtn.disabled = true;
    }
    return isValid ? num : null;
  }

  /**
   * 綁定介面互動事件
   */
  function bindEvents() {
    // 伺服器彈窗控制函式
    function openServerModal() {
      if (dom.serverUrlInput) dom.serverUrlInput.value = state.serverUrl;
      if (dom.serverModalFeedback) dom.serverModalFeedback.textContent = "";
      if (dom.serverModal) dom.serverModal.style.display = "flex";
    }

    function closeServerModal() {
      if (dom.serverModal) dom.serverModal.style.display = "none";
    }

    if (dom.serverConfigBtn) {
      dom.serverConfigBtn.addEventListener("click", openServerModal);
    }
    if (dom.serverModalCloseBtn) {
      dom.serverModalCloseBtn.addEventListener("click", closeServerModal);
    }
    if (dom.serverModal) {
      dom.serverModal.addEventListener("click", (e) => {
        if (e.target === dom.serverModal) closeServerModal();
      });
    }

    // 彈窗內部測試連線按鈕
    if (dom.testServerBtn) {
      dom.testServerBtn.addEventListener("click", async () => {
        dom.testServerBtn.disabled = true;
        dom.testServerBtn.textContent = "探測中...";
        if (dom.serverModalFeedback) dom.serverModalFeedback.textContent = "正在測試連線...";
        const ok = await checkConnection();
        dom.testServerBtn.disabled = false;
        dom.testServerBtn.textContent = "測試連線";
        if (dom.serverModalFeedback) {
          dom.serverModalFeedback.textContent = ok ? "● 連線成功" : "○ 無法連線至伺服器";
          dom.serverModalFeedback.style.color = ok ? "var(--status-online)" : "var(--accent-rose)";
        }
      });
    }

    // 彈窗內部儲存並連線按鈕
    if (dom.saveServerBtn) {
      dom.saveServerBtn.addEventListener("click", async () => {
        const url = dom.serverUrlInput.value.trim().replace(/\/+$/, "");
        localStorage.setItem("kgat_server_url", url);
        state.serverUrl = url;
        const ok = await checkConnection();
        closeServerModal();
        if (ok) executeInference();
      });
    }

    // 頂部狀態標籤點擊開啟伺服器彈窗
    dom.connectionStatus.addEventListener("click", openServerModal);

    // 重新探測按鈕 (向下相容)
    if (dom.reconnectBtn) {
      dom.reconnectBtn.addEventListener("click", async () => {
        await checkConnection();
        executeInference();
      });
    }

    // 背景處理按鈕
    dom.modalBackgroundBtn.addEventListener("click", () => {
      state.isBackgroundMode = true;
      hideInferenceModal();
      showTaskPill();
    });

    // 中斷按鈕
    dom.modalAbortBtn.addEventListener("click", () => {
      abortCurrentOperation();
    });

    // 右上角狀態膠囊點擊（喚回中央遮罩或載入已完成結果）
    dom.taskStatusPill.addEventListener("click", () => {
      if (state.backgroundTask && state.backgroundTask.isCompleted && state.backgroundTask.result) {
        // 背景運算完成，點擊直接載入該筆最新運算結果
        const bgUid = state.backgroundTask.userId;
        const bgDepth = state.backgroundTask.depth;
        state.currentUserId = bgUid;
        state.currentDepth = bgDepth;
        if (dom.userIdInput) {
          dom.userIdInput.value = bgUid;
          validateUserId(bgUid);
        }
        if (dom.userSelect) dom.userSelect.value = bgUid;
        syncCaseByUserId(bgUid);
        dom.depthButtonGroup.querySelectorAll(".btn-group-item").forEach((b) => {
          b.classList.toggle("active", b.dataset.depth === bgDepth);
        });

        const resData = state.backgroundTask.result;
        state.backgroundTask = null;
        state.isBackgroundMode = false;
        hideTaskPill();

        state.currentRecommendData = resData;
        state.selectedPathIndex = 0;
        renderRecommendation(resData);
        renderGraph(resData.explanations);
        setProbeMode("normal");
      } else {
        // 運算進行中，喚回中央聚焦遮罩
        showInferenceModal(dom.modalStageTitle.textContent, dom.modalStageDesc.textContent);
        hideTaskPill();
      }
    });

    // 模型切換
    dom.depthButtonGroup.addEventListener("click", (e) => {
      const btn = e.target.closest(".btn-group-item");
      if (!btn) return;
      dom.depthButtonGroup.querySelectorAll(".btn-group-item").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.currentDepth = btn.dataset.depth;
      executeInference(false);
    });

    function syncCaseByUserId(uid) {
      const matchedCase = state.demoData?.academic_cases?.find((c) => c.user_id === uid);
      if (matchedCase) {
        dom.caseSelect.value = matchedCase.case_id;
        state.currentCaseId = matchedCase.case_id;
      } else {
        dom.caseSelect.value = "";
        state.currentCaseId = "";
      }
    }

    // 典型案例切換
    dom.caseSelect.addEventListener("change", () => {
      const caseId = dom.caseSelect.value;
      if (!caseId) return;
      state.currentCaseId = caseId;
      const targetCase = state.demoData?.academic_cases?.find((c) => c.case_id === caseId);
      if (targetCase) {
        state.currentUserId = targetCase.user_id;
        if (dom.userIdInput) {
          dom.userIdInput.value = targetCase.user_id;
          validateUserId(targetCase.user_id);
        }
        dom.userSelect.value = targetCase.user_id;
        executeInference();
      }
    });

    // 使用者下拉選單切換
    dom.userSelect.addEventListener("change", () => {
      const val = parseInt(dom.userSelect.value, 10);
      state.currentUserId = val;
      if (dom.userIdInput) {
        dom.userIdInput.value = val;
        validateUserId(val);
      }
      syncCaseByUserId(val);
      executeInference();
    });

    // 使用者編號手動輸入框監聽（即時防呆與限制）
    if (dom.userIdInput) {
      dom.userIdInput.addEventListener("input", () => {
        validateUserId(dom.userIdInput.value);
      });

      dom.userIdInput.addEventListener("change", () => {
        const val = validateUserId(dom.userIdInput.value);
        if (val !== null) {
          state.currentUserId = val;
          if (dom.userSelect) dom.userSelect.value = val;
          syncCaseByUserId(val);
          executeInference();
        }
      });

      dom.userIdInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          const val = validateUserId(dom.userIdInput.value);
          if (val !== null) {
            state.currentUserId = val;
            if (dom.userSelect) dom.userSelect.value = val;
            syncCaseByUserId(val);
            executeInference();
          }
        }
      });
    }

    // 執行推論按鈕
    dom.runInferenceBtn.addEventListener("click", () => {
      if (dom.userIdInput) {
        const val = validateUserId(dom.userIdInput.value);
        if (val !== null) {
          state.currentUserId = val;
          if (dom.userSelect) dom.userSelect.value = val;
          syncCaseByUserId(val);
        } else {
          return;
        }
      }
      executeInference(true);
    });

    // 忠實度探針三態切換按鈕
    dom.probeNormalBtn.addEventListener("click", () => setProbeMode("normal"));
    dom.probeFPlusBtn.addEventListener("click", () => setProbeMode("f_plus"));
    dom.probeFMinusBtn.addEventListener("click", () => setProbeMode("f_minus"));
  }

  /**
   * 執行推薦推論（具備自訂結果快取、聚焦遮罩與雙軌背景中斷支援）
   */
  async function executeInference(isExplicitRun = false) {
    const uid = state.currentUserId;
    const depth = state.currentDepth;

    // 0. 優先查核自訂使用者結果快取，杜絕切換模型畫面跳白
    const cachedCustom = state.customUserResults[uid]?.[depth];
    if (cachedCustom) {
      state.currentRecommendData = cachedCustom;
      state.selectedPathIndex = 0;
      renderRecommendation(cachedCustom);
      renderGraph(cachedCustom.explanations);
      setProbeMode("normal");
      return;
    }

    // 0b. 若非點擊「執行推論」，優先檢查離線展示名冊快取（杜絕切換案例或模型時閃爍彈窗）
    if (!isExplicitRun && state.demoData?.user_cache?.[String(uid)]?.depths?.[depth]) {
      const uCache = state.demoData.user_cache[String(uid)];
      const dCache = uCache.depths[depth];
      const metrics = state.demoData.benchmark_metrics?.[depth] || {};
      const recData = {
        user_id: uid,
        user_name: uCache.user_name,
        depth: depth,
        recommended_item_id: dCache.recommended_item_id,
        recommended_item_name: dCache.recommended_item_name,
        score: dCache.score,
        prob: dCache.prob,
        is_test_hit: dCache.is_test_hit,
        user_n_interactions: dCache.user_n_interactions,
        item_kg_degree: dCache.item_kg_degree,
        metrics: {
          hr_20: metrics.hr_20 || 0,
          ndcg_20: metrics.ndcg_20 || 0,
        },
        explanations: dCache.explanations || [],
        fidelity: dCache.fidelity || {},
      };
      state.currentRecommendData = recData;
      state.selectedPathIndex = 0;
      renderRecommendation(recData);
      renderGraph(recData.explanations);
      setProbeMode("normal");
      return;
    }

    let recData = null;

    // 1. 若處於連線狀態，透過伺服器進行推論
    if (state.isConnected) {
      state.isComputing = true;
      state.activeAbortController = new AbortController();
      state.currentTaskLabel = `● 推論中：使用者 ${uid}`;
      const controller = state.activeAbortController;

      state.backgroundTask = {
        userId: uid,
        depth: depth,
        controller: controller,
        result: null,
        isCompleted: false,
        label: state.currentTaskLabel,
      };

      if (!state.isBackgroundMode) {
        showInferenceModal("推論計算中", `正在為使用者 ${uid} 進行模型 ${depth} 評估...`);
      } else {
        showTaskPill(`● 推論中：使用者 ${uid}`);
      }

      dom.runInferenceBtn.classList.add("btn-loading");
      dom.runInferenceBtn.textContent = "推論中...";

      try {
        const res = await fetch(`${state.serverUrl}/api/recommend`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ user_id: uid, depth: depth }),
          signal: controller.signal,
        });

        if (res.ok) {
          recData = await res.json();
          // 即時保存至自訂使用者快取字典
          if (!state.customUserResults[uid]) {
            state.customUserResults[uid] = {};
          }
          state.customUserResults[uid][depth] = recData;
        }
      } catch (err) {
        if (err.name === "AbortError") {
          console.log("使用者已手動中斷推論運算");
          abortCurrentOperation();
          return;
        }
        console.warn("後端推論異常，自動降級至離線資料，不更動連線狀態：", err);
      } finally {
        state.isComputing = false;
        state.activeAbortController = null;
        dom.runInferenceBtn.classList.remove("btn-loading");
        dom.runInferenceBtn.textContent = "執行推論";

        if (state.isBackgroundMode) {
          if (recData) {
            state.backgroundTask.result = recData;
            state.backgroundTask.isCompleted = true;
            // 檢查使用者是否仍停留在該筆使用者與模型頁面
            if (state.currentUserId === uid && state.currentDepth === depth) {
              hideInferenceModal();
              hideTaskPill();
              state.isBackgroundMode = false;
              state.backgroundTask = null;
            } else {
              // 使用者已切換至其他案例，膠囊轉為就緒標記，不卡死或覆寫畫面
              showTaskPill(`● 完成：使用者 ${uid}（點擊檢視）`, true);
              return;
            }
          } else {
            hideInferenceModal();
            hideTaskPill();
            state.isBackgroundMode = false;
            state.backgroundTask = null;
          }
        } else {
          hideInferenceModal();
          hideTaskPill();
          state.backgroundTask = null;
        }
      }
    }

    // 2. 離線快取回退方案
    if (!recData && state.demoData) {
      const uCache = state.demoData.user_cache?.[String(uid)];
      const dCache = uCache?.depths?.[depth];
      const metrics = state.demoData.benchmark_metrics?.[depth] || {};

      if (dCache) {
        recData = {
          user_id: uid,
          user_name: uCache.user_name,
          depth: depth,
          recommended_item_id: dCache.recommended_item_id,
          recommended_item_name: dCache.recommended_item_name,
          score: dCache.score,
          prob: dCache.prob,
          is_test_hit: dCache.is_test_hit,
          user_n_interactions: dCache.user_n_interactions,
          item_kg_degree: dCache.item_kg_degree,
          metrics: {
            hr_20: metrics.hr_20 || 0,
            ndcg_20: metrics.ndcg_20 || 0,
          },
          explanations: dCache.explanations || [],
          fidelity: dCache.fidelity || {},
        };
      }
    }

    if (!recData) {
      recData = {
        user_id: uid,
        user_name: `使用者 ${uid}`,
        depth: depth,
        recommended_item_id: 0,
        recommended_item_name: "無推論資料",
        score: 0,
        prob: 0,
        is_test_hit: false,
        user_n_interactions: 0,
        item_kg_degree: 0,
        metrics: { hr_20: 0, ndcg_20: 0 },
        explanations: [],
        fidelity: {},
      };
    }

    state.currentRecommendData = recData;
    state.selectedPathIndex = 0;

    // 渲染推薦結果
    renderRecommendation(recData);

    // 渲染子圖
    renderGraph(recData.explanations);

    // 重設並渲染忠實度分析
    setProbeMode("normal");
  }

  /**
   * 渲染推薦結果區塊
   */
  function renderRecommendation(data) {
    dom.recipeName.textContent = data.recommended_item_name || "未知食譜";
    dom.recipeId.textContent = data.recommended_item_id || "-";

    const prob = data.prob || 0;
    dom.origProbVal.textContent = prob.toFixed(4);
    if (dom.origProbBaseFill) {
      dom.origProbBaseFill.style.width = `${Math.min(100, prob * 100).toFixed(1)}%`;
    }
    if (dom.origProbFill) {
      dom.origProbFill.style.width = `${Math.min(100, prob * 100).toFixed(1)}%`;
      dom.origProbFill.classList.remove("dim");
    }

    if (dom.testHitBadge) {
      dom.testHitBadge.style.display = "none";
    }

    dom.userInteractionsVal.textContent = data.user_n_interactions || 0;
    dom.itemDegreeVal.textContent = data.item_kg_degree || 0;

    const hr = data.metrics?.hr_20 || 0;
    const ndcg = data.metrics?.ndcg_20 || 0;
    dom.modelHrVal.textContent = hr.toFixed(4);
    dom.modelNdcgVal.textContent = ndcg.toFixed(4);

    // 渲染案例分析說明
    const currentCase = state.demoData?.academic_cases?.find((c) => c.user_id === data.user_id);
    if (currentCase) {
      dom.caseInsightBox.style.display = "block";
      dom.caseInsightTitle.textContent = currentCase.name;
      dom.caseInsightDesc.textContent = currentCase.description;
    } else {
      dom.caseInsightBox.style.display = "none";
    }
  }

  /**
   * 執行探針模式切換與忠實度分析
   */
  async function setProbeMode(mode) {
    state.currentProbeMode = mode;

    [dom.probeNormalBtn, dom.probeFPlusBtn, dom.probeFMinusBtn].forEach((btn) => {
      if (btn.dataset.mode === mode) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    });

    const recData = state.currentRecommendData;
    if (!recData) return;

    const isNonAttentionModel = !["3", "2", "1"].includes(state.currentDepth);
    const hasNoExplanations = !recData.explanations || recData.explanations.length === 0;

    // 非圖注意力模型或無路徑使用者防護：整體反灰並禁用點擊
    if (isNonAttentionModel) {
      dom.fidelitySection.classList.add("disabled");
      dom.noPathBadge.style.display = "inline-block";
      dom.noPathBadge.textContent = "此模型不具備圖注意力路徑";
      if (dom.graphSection) dom.graphSection.classList.add("disabled");

      dom.barOrigProb.style.width = `${Math.min(100, (recData.prob || 0) * 100).toFixed(1)}%`;
      dom.barOrigText.textContent = (recData.prob || 0).toFixed(4);
      dom.barProbeProb.style.width = "0%";
      dom.barProbeText.textContent = "-";
      dom.fPlusVal.textContent = "-";
      dom.fMinusVal.textContent = "-";
      dom.cardFPlus.classList.remove("highlight");
      dom.cardFMinus.classList.remove("highlight");
      if (dom.promptFPlus) dom.promptFPlus.style.display = "none";
      if (dom.promptFMinus) dom.promptFMinus.style.display = "none";
      if (dom.promptNormal) dom.promptNormal.style.display = "block";
      updateGraphMaskingState("normal");
      return;
    }

    if (hasNoExplanations) {
      dom.fidelitySection.classList.add("disabled");
      dom.noPathBadge.style.display = "inline-block";
      dom.noPathBadge.textContent = "無路徑使用者";
      if (dom.graphSection) dom.graphSection.classList.remove("disabled");

      dom.barOrigProb.style.width = `${Math.min(100, (recData.prob || 0) * 100).toFixed(1)}%`;
      dom.barOrigText.textContent = (recData.prob || 0).toFixed(4);
      dom.barProbeProb.style.width = "0%";
      dom.barProbeText.textContent = "-";
      dom.fPlusVal.textContent = "-";
      dom.fMinusVal.textContent = "-";
      dom.cardFPlus.classList.remove("highlight");
      dom.cardFMinus.classList.remove("highlight");
      if (dom.promptFPlus) dom.promptFPlus.style.display = "none";
      if (dom.promptFMinus) dom.promptFMinus.style.display = "none";
      if (dom.promptNormal) dom.promptNormal.style.display = "block";
      updateGraphMaskingState("normal");
      return;
    }

    // 有路徑之圖注意力模型：恢復忠實度面板可用狀態
    dom.fidelitySection.classList.remove("disabled");
    dom.noPathBadge.style.display = "none";
    if (dom.graphSection) dom.graphSection.classList.remove("disabled");

    const origProb = recData.prob || 0;
    let probeProb = origProb;
    let fidelityScore = 0.0;

    // 取得當前選定路徑節點以供探針執行精確遮蔽
    const currentExp = recData.explanations?.[state.selectedPathIndex];
    let pathNodes = null;
    if (currentExp && currentExp.path_details && currentExp.path_details.length > 0) {
      pathNodes = [currentExp.path_details.map((d) => d.id_remapped)];
    }

    // 優先向後端發出探針請求
    if (state.isConnected && recData.recommended_item_id && mode !== "normal") {
      state.isComputing = true;
      state.activeAbortController = new AbortController();
      const modeText = mode === "f_plus" ? "必要性忠實度" : "充分性忠實度";
      state.currentTaskLabel = `● 忠實度分析中：使用者 ${recData.user_id}`;

      showInferenceModal("忠實度分析中", `正在針對選定路徑計算${modeText}...`);

      try {
        const res = await fetch(`${state.serverUrl}/api/fidelity_probe`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: recData.user_id,
            item_id: recData.recommended_item_id,
            depth: state.currentDepth,
            mode: mode,
            path_nodes: pathNodes,
          }),
          signal: state.activeAbortController.signal,
        });

        if (res.ok) {
          const probeRes = await res.json();
          probeProb = probeRes.prob;
          fidelityScore = probeRes.fidelity_score;
        }
      } catch (e) {
        if (e.name === "AbortError") {
          console.log("忠實度探針計算已中斷");
          abortCurrentOperation();
          return;
        }
        console.warn("探針請求失敗，回退至快取數據");
      } finally {
        state.isComputing = false;
        state.activeAbortController = null;
        hideInferenceModal();
        hideTaskPill();
      }
    } else {
      // 離線快取探針數值
      const fid = recData.fidelity || {};
      if (mode === "f_plus") {
        probeProb = fid.prob_f_plus !== undefined && fid.prob_f_plus !== null ? fid.prob_f_plus : origProb;
        fidelityScore = fid.f_plus !== undefined && fid.f_plus !== null ? fid.f_plus : 0.0;
      } else if (mode === "f_minus") {
        probeProb = fid.prob_f_minus !== undefined && fid.prob_f_minus !== null ? fid.prob_f_minus : origProb;
        fidelityScore = fid.f_minus !== undefined && fid.f_minus !== null ? fid.f_minus : 0.0;
      }
    }

    // 計算完成後寫入快取，確保跨模式切換時數值持久保留
    if (!recData.fidelity) recData.fidelity = {};
    if (mode === "f_plus") {
      recData.fidelity.f_plus = fidelityScore;
      recData.fidelity.prob_f_plus = probeProb;
    } else if (mode === "f_minus") {
      recData.fidelity.f_minus = fidelityScore;
      recData.fidelity.prob_f_minus = probeProb;
    }

    // 忠實度面板長條更新
    dom.barOrigProb.style.width = `${Math.min(100, origProb * 100).toFixed(1)}%`;
    dom.barOrigText.textContent = origProb.toFixed(4);

    dom.barProbeProb.style.width = `${Math.min(100, probeProb * 100).toFixed(1)}%`;
    dom.barProbeText.textContent = probeProb.toFixed(4);
    if (mode === "f_plus") {
      dom.barProbeProb.className = "waterlevel-bar-fill fill-probe dim";
    } else {
      dom.barProbeProb.className = "waterlevel-bar-fill fill-probe";
    }

    // 預測機率卡片雙層疊加進度條與數值更新
    if (dom.origProbBaseFill) {
      dom.origProbBaseFill.style.width = `${Math.min(100, origProb * 100).toFixed(1)}%`;
    }
    if (dom.origProbFill) {
      dom.origProbFill.style.width = `${Math.min(100, probeProb * 100).toFixed(1)}%`;
      if (mode === "f_plus") {
        dom.origProbFill.classList.add("dim");
      } else {
        dom.origProbFill.classList.remove("dim");
      }
    }
    if (dom.origProbVal) {
      dom.origProbVal.textContent = probeProb.toFixed(4);
    }

    // 更新 F+ 與 F- 數值卡片 (持續保留，切換模式不重置為 -)
    const fPlusVal = recData.fidelity?.f_plus;
    const fMinusVal = recData.fidelity?.f_minus;

    if (fPlusVal !== undefined && fPlusVal !== null) {
      dom.fPlusVal.textContent = typeof fPlusVal === "number" ? fPlusVal.toFixed(4) : fPlusVal;
    } else {
      dom.fPlusVal.textContent = "-";
    }
    dom.cardFPlus.classList.toggle("highlight", mode === "f_plus");

    if (fMinusVal !== undefined && fMinusVal !== null) {
      dom.fMinusVal.textContent = typeof fMinusVal === "number" ? fMinusVal.toFixed(4) : fMinusVal;
    } else {
      dom.fMinusVal.textContent = "-";
    }
    dom.cardFMinus.classList.toggle("highlight", mode === "f_minus");

    // 依當前模式動態切換顯示單一斜體提示
    if (dom.promptFPlus && dom.promptFMinus) {
      if (mode === "f_plus") {
        if (dom.promptNormal) dom.promptNormal.style.display = "none";
        dom.promptFMinus.style.display = "none";
        dom.promptFPlus.style.display = "block";
      } else if (mode === "f_minus") {
        if (dom.promptNormal) dom.promptNormal.style.display = "none";
        dom.promptFPlus.style.display = "none";
        dom.promptFMinus.style.display = "block";
      } else {
        dom.promptFPlus.style.display = "none";
        dom.promptFMinus.style.display = "none";
        if (dom.promptNormal) dom.promptNormal.style.display = "block";
      }
    }

    // 更新子圖連線視覺狀態
    updateGraphMaskingState(mode);
  }

  /**
   * 繪製背景裝飾實體點與虛擬連線，增進子圖空間感
   */
  function renderBackgroundDeco() {
    if (!dom.svgBgDecoGroup) return;
    dom.svgBgDecoGroup.innerHTML = "";

    const decoPoints = [
      { x: 60, y: 70 }, { x: 170, y: 48 }, { x: 280, y: 75 }, { x: 390, y: 42 },
      { x: 510, y: 65 }, { x: 630, y: 50 }, { x: 740, y: 80 }, { x: 50, y: 390 },
      { x: 170, y: 415 }, { x: 290, y: 395 }, { x: 420, y: 420 }, { x: 560, y: 405 },
      { x: 710, y: 390 }, { x: 110, y: 145 }, { x: 690, y: 155 }, { x: 115, y: 305 },
      { x: 685, y: 315 }
    ];

    const decoEdges = [
      [0, 1], [1, 2], [3, 4], [4, 5], [5, 6],
      [7, 8], [8, 9], [9, 10], [10, 11], [11, 12],
      [0, 13], [6, 14], [7, 15], [12, 16]
    ];

    decoEdges.forEach(([i, j]) => {
      const p1 = decoPoints[i];
      const p2 = decoPoints[j];
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", p1.x);
      line.setAttribute("y1", p1.y);
      line.setAttribute("x2", p2.x);
      line.setAttribute("y2", p2.y);
      line.setAttribute("class", "svg-bg-deco-line");
      dom.svgBgDecoGroup.appendChild(line);
    });

    decoPoints.forEach((pt) => {
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", pt.x);
      circle.setAttribute("cy", pt.y);
      circle.setAttribute("r", "5");
      circle.setAttribute("class", "svg-bg-deco-node");
      dom.svgBgDecoGroup.appendChild(circle);
    });
  }

  /**
   * 根據探針模式更新子圖連線視覺狀態
   */
  function updateGraphMaskingState(mode) {
    const activePathIdx = state.selectedPathIndex;

    // 背景裝飾點：遮蔽其餘路徑時淡出消失，凸顯核心路徑之孤立充分性
    if (dom.svgBgDecoGroup) {
      dom.svgBgDecoGroup.classList.toggle("faded", mode === "f_minus");
    }

    // 邊線視覺狀態更新
    const edges = dom.svgEdgesGroup.querySelectorAll(".svg-edge");
    edges.forEach((edge) => {
      const pathAttr = edge.getAttribute("data-paths") || "";
      const pathIndices = pathAttr.split(",").filter((s) => s.length > 0).map(Number);
      const isPrimary = pathIndices.includes(activePathIdx);
      const isSecondary = pathIndices.some((idx) => idx !== activePathIdx);

      edge.classList.remove("masked", "backup-active", "faded");

      if (mode === "normal") {
        if (isPrimary) {
          edge.className.baseVal = "svg-edge path-primary";
          edge.setAttribute("marker-end", "url(#arrow-active)");
        } else {
          edge.className.baseVal = "svg-edge path-secondary";
          edge.setAttribute("marker-end", "url(#arrow)");
        }
      } else if (mode === "f_plus") {
        // 遮蔽此路徑 (F+)：主要路徑淡出，點亮次要/備援解釋路徑
        if (isPrimary && !isSecondary) {
          edge.className.baseVal = "svg-edge path-primary masked";
          edge.setAttribute("marker-end", "url(#arrow)");
        } else if (isSecondary) {
          edge.className.baseVal = "svg-edge path-secondary backup-active";
          edge.setAttribute("marker-end", "url(#arrow-backup)");
        }
      } else if (mode === "f_minus") {
        // 遮蔽其餘路徑 (F-)：次要路徑淡出，主要路徑孤立充分性凸顯
        if (isPrimary) {
          edge.className.baseVal = "svg-edge path-primary";
          edge.setAttribute("marker-end", "url(#arrow-active)");
        } else {
          edge.className.baseVal = "svg-edge path-secondary faded";
          edge.setAttribute("marker-end", "url(#arrow)");
        }
      }
    });

    // 節點視覺狀態更新
    const nodes = dom.svgNodesGroup.querySelectorAll(".svg-node");
    nodes.forEach((nodeEl) => {
      const pathAttr = nodeEl.getAttribute("data-paths") || "";
      const pathIndices = pathAttr.split(",").filter((s) => s.length > 0).map(Number);
      const isPrimary = pathIndices.includes(activePathIdx);
      const isSecondary = pathIndices.some((idx) => idx !== activePathIdx);
      const nodeType = nodeEl.getAttribute("data-type") || "";

      nodeEl.classList.remove("node-masked", "node-backup-active");

      if (mode === "f_plus") {
        if (isPrimary && !isSecondary && nodeType !== "USER" && nodeType !== "RECIPE_TARGET") {
          nodeEl.classList.add("node-masked");
        } else if (isSecondary && nodeType !== "USER" && nodeType !== "RECIPE_TARGET") {
          nodeEl.classList.add("node-backup-active");
        }
      } else if (mode === "f_minus") {
        if (!isPrimary && nodeType !== "USER" && nodeType !== "RECIPE_TARGET") {
          nodeEl.classList.add("node-masked");
        }
      }
    });
  }

  /**
   * 繪製協同知識圖注意力子圖 (SVG)
   */
  function renderGraph(explanations) {
    dom.svgEdgesGroup.innerHTML = "";
    dom.svgNodesGroup.innerHTML = "";
    dom.pathsSelector.innerHTML = "";

    const isNonAttention = !["3", "2", "1"].includes(state.currentDepth);

    if (!explanations || explanations.length === 0) {
      dom.graphEmptyTitle.textContent = isNonAttention ? "非圖注意力模型" : "無可達解釋路徑";
      dom.graphEmptyDesc.textContent = isNonAttention
        ? "此模型不具備圖注意力路徑。"
        : "使用者節點於當前傳播深度下無相連路徑。";
      dom.graphEmptyMsg.style.display = "block";
      if (dom.graphSection) dom.graphSection.classList.toggle("disabled", isNonAttention);
      if (dom.svgBgDecoGroup) dom.svgBgDecoGroup.innerHTML = "";
      return;
    }

    dom.graphEmptyMsg.style.display = "none";
    if (dom.graphSection) dom.graphSection.classList.remove("disabled");

    // 繪製背景裝飾節點
    renderBackgroundDeco();

    // 渲染路徑選單 (路徑聯合分數 π 於此統一標示)
    explanations.forEach((exp, idx) => {
      const item = document.createElement("div");
      item.className = `path-selector-item ${idx === state.selectedPathIndex ? "selected" : ""}`;
      item.innerHTML = `
        <span class="path-desc">路徑 ${idx + 1}：${exp.path_description}</span>
        <span class="path-weight">π = ${typeof exp.contribution_score === "number" ? exp.contribution_score.toExponential(3) : "0.000e+0"}</span>
      `;
      item.addEventListener("click", () => {
        state.selectedPathIndex = idx;
        dom.pathsSelector.querySelectorAll(".path-selector-item").forEach((el, i) => {
          el.classList.toggle("selected", i === idx);
        });
        drawConvergenceGraph(explanations);
        updateGraphMaskingState(state.currentProbeMode);
      });
      dom.pathsSelector.appendChild(item);
    });

    // 繪製多交互匯聚子圖 (重現 Case C 匯聚拓撲網絡)
    drawConvergenceGraph(explanations);
    updateGraphMaskingState(state.currentProbeMode);
  }

  /**
   * 繪製多交互匯聚子圖（重現論文 Case C 拓撲結構）
   */
  function drawConvergenceGraph(explanations) {
    dom.svgEdgesGroup.innerHTML = "";
    dom.svgNodesGroup.innerHTML = "";

    const topPaths = explanations.slice(0, 3);
    if (topPaths.length === 0) return;

    const svgWidth = 800;
    const svgHeight = 440;
    const paddingX = 90;
    const paddingY = 60;
    const centerY = svgHeight / 2;

    const maxHops = Math.max(...topPaths.map((p) => (p.path_details || []).length - 1), 1);

    // 1. 節點去重與層級索引分類
    const uniqueNodes = new Map();
    topPaths.forEach((pathObj, pIdx) => {
      const details = pathObj.path_details || [];
      const pathLen = details.length;
      details.forEach((node, hopIdx) => {
        const nid = node.id_remapped;
        const normalizedHop = hopIdx === pathLen - 1 ? maxHops : hopIdx;
        if (!uniqueNodes.has(nid)) {
          uniqueNodes.set(nid, {
            ...node,
            hop: normalizedHop,
            paths: new Set([pIdx]),
          });
        } else {
          uniqueNodes.get(nid).paths.add(pIdx);
        }
      });
    });

    // 依 hop 分組計算階層坐標
    const hopGroups = Array.from({ length: maxHops + 1 }, () => []);
    uniqueNodes.forEach((nodeObj) => {
      const h = Math.min(nodeObj.hop, maxHops);
      hopGroups[h].push(nodeObj);
    });

    const nodeCoords = new Map();
    hopGroups.forEach((group, h) => {
      const count = group.length;
      const x = paddingX + (h / maxHops) * (svgWidth - 2 * paddingX);
      group.forEach((nodeObj, idx) => {
        let y = centerY;
        if (count > 1) {
          const spacingY = Math.min(110, (svgHeight - 2 * paddingY) / Math.max(1, count - 1));
          y = centerY + (idx - (count - 1) / 2) * spacingY;
        }
        nodeCoords.set(nodeObj.id_remapped, { x, y, node: nodeObj });
      });
    });

    // 2. 邊緣去重與繪製 (邊線不再重複印製 π 分數，維持純淨線條)
    const edgeMap = new Map();
    topPaths.forEach((pathObj, pIdx) => {
      const details = pathObj.path_details || [];
      for (let s = 0; s < details.length - 1; s++) {
        const fromId = details[s].id_remapped;
        const toId = details[s + 1].id_remapped;
        const edgeKey = `${fromId}_${toId}`;
        if (!edgeMap.has(edgeKey)) {
          edgeMap.set(edgeKey, { fromId, toId, paths: new Set([pIdx]) });
        } else {
          edgeMap.get(edgeKey).paths.add(pIdx);
        }
      }
    });

    const allScores = explanations.map((e) => e.contribution_score || 0);
    const maxScore = Math.max(...allScores, 1e-6);
    const minScore = Math.min(...allScores, 0);
    const scoreRange = Math.max(maxScore - minScore, 1e-6);

    edgeMap.forEach((edgeObj) => {
      const c1 = nodeCoords.get(edgeObj.fromId);
      const c2 = nodeCoords.get(edgeObj.toId);
      if (!c1 || !c2) return;

      const isPrimary = edgeObj.paths.has(state.selectedPathIndex);
      const selectedExp = explanations[state.selectedPathIndex] || explanations[0];
      const normScore = Math.max(0, Math.min(1, ((selectedExp?.contribution_score || 0) - minScore) / scoreRange));
      const dynamicStrokeWidth = isPrimary ? (3.2 + normScore * 2.2).toFixed(1) : "2.0";

      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", c1.x);
      line.setAttribute("y1", c1.y);
      line.setAttribute("x2", c2.x);
      line.setAttribute("y2", c2.y);
      line.setAttribute("data-from", edgeObj.fromId);
      line.setAttribute("data-to", edgeObj.toId);
      line.setAttribute("data-paths", Array.from(edgeObj.paths).join(","));
      line.setAttribute("stroke-width", dynamicStrokeWidth);
      line.setAttribute("class", isPrimary ? "svg-edge path-primary" : "svg-edge path-secondary");
      line.setAttribute("marker-end", isPrimary ? "url(#arrow-active)" : "url(#arrow)");
      dom.svgEdgesGroup.appendChild(line);
    });

    // 3. 繪製去重後之網路節點
    nodeCoords.forEach((coord) => {
      const n = coord.node;
      const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
      group.setAttribute("class", "svg-node");
      group.setAttribute("data-id", n.id_remapped);
      group.setAttribute("data-paths", Array.from(n.paths).join(","));
      group.setAttribute("data-type", n.hop === maxHops ? "RECIPE_TARGET" : n.type);
      group.setAttribute("transform", `translate(${coord.x}, ${coord.y})`);

      let fillColor = "#2563eb"; // USER
      let typeLabel = "使用者";
      if (n.type === "RECIPE") {
        fillColor = "#d97706";
        typeLabel = n.hop === maxHops ? "推薦目標" : "歷史食譜";
      } else if (n.type === "INGREDIENT") {
        fillColor = "#059669";
        typeLabel = "食材";
      } else if (n.type === "TAG") {
        fillColor = "#7c3aed";
        typeLabel = "標籤";
      }

      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("r", "20");
      circle.setAttribute("fill", fillColor);
      group.appendChild(circle);

      const nameText = document.createElementNS("http://www.w3.org/2000/svg", "text");
      nameText.setAttribute("class", "svg-node-label");
      nameText.setAttribute("y", "36");
      const shortName = n.name.length > 16 ? n.name.substring(0, 15) + "..." : n.name;
      nameText.textContent = shortName;
      group.appendChild(nameText);

      const typeText = document.createElementNS("http://www.w3.org/2000/svg", "text");
      typeText.setAttribute("class", "svg-node-type");
      typeText.setAttribute("y", "-26");
      typeText.textContent = typeLabel;
      group.appendChild(typeText);

      dom.svgNodesGroup.appendChild(group);
    });
  }

  // 頁面載入完成後啟動
  window.addEventListener("DOMContentLoaded", init);
})();
