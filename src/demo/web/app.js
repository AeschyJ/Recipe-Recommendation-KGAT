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

    graphSection: document.getElementById("graphSection"),
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
    const url = dom.serverUrlInput.value.trim().replace(/\/+$/, "");
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
    // 重新探測按鈕與連線狀態標籤點擊
    dom.reconnectBtn.addEventListener("click", async () => {
      await checkConnection();
      executeInference();
    });
    dom.connectionStatus.addEventListener("click", async () => {
      await checkConnection();
      executeInference();
    });

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
    dom.origProbFill.style.width = `${Math.min(100, prob * 100).toFixed(1)}%`;

    if (data.is_test_hit) {
      dom.testHitBadge.className = "badge-hit yes";
      dom.testHitBadge.textContent = "命中測試集";
    } else {
      dom.testHitBadge.className = "badge-hit no";
      dom.testHitBadge.textContent = "未命中測試集";
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

    // 水位長條更新
    dom.barOrigProb.style.width = `${Math.min(100, origProb * 100).toFixed(1)}%`;
    dom.barOrigText.textContent = origProb.toFixed(4);

    dom.barProbeProb.style.width = `${Math.min(100, probeProb * 100).toFixed(1)}%`;
    dom.barProbeText.textContent = probeProb.toFixed(4);
    if (mode === "f_plus") {
      dom.barProbeProb.className = "waterlevel-bar-fill fill-probe dim";
    } else {
      dom.barProbeProb.className = "waterlevel-bar-fill fill-probe";
    }

    // 更新 F+ 與 F- 數值卡片
    const fPlusCached = recData.fidelity?.f_plus;
    const fMinusCached = recData.fidelity?.f_minus;

    if (fPlusCached !== undefined && fPlusCached !== null) {
      dom.fPlusVal.textContent = fPlusCached.toFixed(4);
      dom.cardFPlus.classList.toggle("highlight", mode === "f_plus");
    } else {
      dom.fPlusVal.textContent = mode === "f_plus" ? fidelityScore.toFixed(4) : "-";
      dom.cardFPlus.classList.toggle("highlight", mode === "f_plus");
    }

    if (fMinusCached !== undefined && fMinusCached !== null) {
      dom.fMinusVal.textContent = fMinusCached.toFixed(4);
      dom.cardFMinus.classList.toggle("highlight", mode === "f_minus");
    } else {
      dom.fMinusVal.textContent = mode === "f_minus" ? fidelityScore.toFixed(4) : "-";
      dom.cardFMinus.classList.toggle("highlight", mode === "f_minus");
    }

    // 更新子圖連線視覺狀態
    updateGraphMaskingState(mode);
  }

  /**
   * 根據探針模式更新子圖連線視覺狀態
   */
  function updateGraphMaskingState(mode) {
    const edges = dom.svgEdgesGroup.querySelectorAll(".svg-edge");
    edges.forEach((edge) => {
      edge.classList.remove("masked", "highlighted");
      if (mode === "f_plus") {
        edge.classList.add("masked");
      } else if (mode === "f_minus") {
        edge.classList.add("highlighted");
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
      return;
    }

    dom.graphEmptyMsg.style.display = "none";
    if (dom.graphSection) dom.graphSection.classList.remove("disabled");

    // 渲染路徑選單
    explanations.forEach((exp, idx) => {
      const item = document.createElement("div");
      item.className = `path-selector-item ${idx === state.selectedPathIndex ? "selected" : ""}`;
      item.innerHTML = `
        <span class="path-desc">路徑 ${idx + 1}：${exp.path_description}</span>
        <span class="path-weight">π = ${exp.contribution_score.toExponential(3)}</span>
      `;
      item.addEventListener("click", () => {
        state.selectedPathIndex = idx;
        dom.pathsSelector.querySelectorAll(".path-selector-item").forEach((el, i) => {
          el.classList.toggle("selected", i === idx);
        });
        drawSinglePath(exp);
      });
      dom.pathsSelector.appendChild(item);
    });

    // 預設繪製第一條或選定路徑
    drawSinglePath(explanations[state.selectedPathIndex] || explanations[0]);
  }

  /**
   * 繪製單條解釋路徑拓撲圖
   */
  function drawSinglePath(exp) {
    dom.svgEdgesGroup.innerHTML = "";
    dom.svgNodesGroup.innerHTML = "";

    const details = exp.path_details || [];
    if (details.length === 0) return;

    const svgWidth = 800;
    const svgHeight = 440;
    const paddingX = 90;
    const centerY = svgHeight / 2;

    const nNodes = details.length;
    const stepX = (svgWidth - 2 * paddingX) / Math.max(1, nNodes - 1);

    // 計算各節點座標
    const nodeCoords = details.map((n, i) => {
      let offsetY = 0;
      if (nNodes > 2) {
        if (i === 1) offsetY = -40;
        if (i === 2) offsetY = 40;
      }
      return {
        x: paddingX + i * stepX,
        y: centerY + offsetY,
        node: n,
      };
    });

    // 1. 繪製邊線與動態注意力權重寬度映射 (2.0px ~ 6.0px)
    const allScores = (state.currentRecommendData?.explanations || []).map((e) => e.contribution_score || 0);
    const maxScore = Math.max(...allScores, 1e-6);
    const minScore = Math.min(...allScores, 0);
    const scoreRange = Math.max(maxScore - minScore, 1e-6);
    const normScore = Math.max(0, Math.min(1, ((exp.contribution_score || 0) - minScore) / scoreRange));
    const dynamicStrokeWidth = (2.0 + normScore * 4.0).toFixed(1);
    const weightText = (exp.contribution_score || 0).toExponential(2);

    for (let i = 0; i < nodeCoords.length - 1; i++) {
      const c1 = nodeCoords[i];
      const c2 = nodeCoords[i + 1];

      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", c1.x);
      line.setAttribute("y1", c1.y);
      line.setAttribute("x2", c2.x);
      line.setAttribute("y2", c2.y);
      line.setAttribute("class", "svg-edge");
      line.setAttribute("stroke-width", dynamicStrokeWidth);
      line.setAttribute("marker-end", "url(#arrow)");
      dom.svgEdgesGroup.appendChild(line);

      // 連線文字（標註注意力權重數值）
      const midX = (c1.x + c2.x) / 2;
      const midY = (c1.y + c2.y) / 2 - 12;
      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", midX);
      text.setAttribute("y", midY);
      text.setAttribute("class", "svg-edge-label");
      text.setAttribute("text-anchor", "middle");
      text.textContent = `π = ${weightText}`;
      dom.svgEdgesGroup.appendChild(text);
    }

    // 2. 繪製節點
    nodeCoords.forEach((coord, idx) => {
      const n = coord.node;
      const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
      group.setAttribute("class", "svg-node");
      group.setAttribute("transform", `translate(${coord.x}, ${coord.y})`);

      // 顏色依型態映射
      let fillColor = "#2563eb"; // USER
      let typeLabel = "使用者";
      if (n.type === "RECIPE") {
        fillColor = "#d97706";
        typeLabel = "食譜";
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

      // 節點名稱
      const nameText = document.createElementNS("http://www.w3.org/2000/svg", "text");
      nameText.setAttribute("class", "svg-node-label");
      nameText.setAttribute("y", "36");
      const shortName = n.name.length > 16 ? n.name.substring(0, 15) + "..." : n.name;
      nameText.textContent = shortName;
      group.appendChild(nameText);

      // 節點類型標籤
      const typeText = document.createElementNS("http://www.w3.org/2000/svg", "text");
      typeText.setAttribute("class", "svg-node-type");
      typeText.setAttribute("y", "-26");
      typeText.textContent = typeLabel;
      group.appendChild(typeText);

      dom.svgNodesGroup.appendChild(group);
    });

    // 根據當前探針模式重繪連線遮蔽狀態
    updateGraphMaskingState(state.currentProbeMode);
  }

  // 頁面載入完成後啟動
  window.addEventListener("DOMContentLoaded", init);
})();
