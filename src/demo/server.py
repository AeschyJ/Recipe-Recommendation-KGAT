"""後端推論與圖遮蔽伺服器.

基於 FastAPI 與 Uvicorn，提供口試現場展示系統之模型推論、
注意力路徑萃取，以及 F+ 與 F- 圖遮蔽探針計算。
預設監聽 8501 端口，嚴禁佔用 8000 端口。
"""

import argparse
import gc
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# 配置 Matplotlib 快取目錄以加快載入
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl_cache"))

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# 配置伺服器輪替日誌
log_dir = os.path.join(project_root, "output", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "demo_server.log")

logger = logging.getLogger("demo_server")
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    )
    logger.addHandler(file_handler)

from src.generate_explanations import get_node_name
from src.model.bpr_mf import BPRMF
from src.model.explainer_attention import KGATAttentionExplainer
from src.model.kgat import KGATAttention, KGAT_BiInteraction
from src.model.lightgcn import LightGCN
from src.model.nfm import NFM
from src.train import construct_adj, get_adj_indices
from src.train_baseline import get_user_item_indices


def fast_load_names_and_maps(data_dir, raw_data_dir):
    """快速載入節點與食譜名稱對應資訊."""
    stats_path = os.path.join(data_dir, "stats.pkl")
    if not os.path.exists(stats_path):
        return None, None, None, {}
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    user_le = stats.get("user_map")
    item_le = stats.get("item_map")
    id_to_ing = {v: k for k, v in stats.get("ingredient_map", {}).items()}
    id_to_tag = {v: k for k, v in stats.get("tag_map", {}).items()}

    recipes_path = os.path.join(raw_data_dir, "RAW_recipes.csv")
    recipe_name_map = {}
    if os.path.exists(recipes_path):
        df_recipes = pd.read_csv(recipes_path, usecols=["id", "name"])
        recipe_name_map = dict(zip(df_recipes["id"], df_recipes["name"]))
    return user_le, item_le, (id_to_ing, id_to_tag), recipe_name_map

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.initialize()
    yield


app = FastAPI(
    title="協同知識圖注意力推薦與忠實度驗證系統",
    description="碩士學位口試現場展示推論與圖遮蔽伺服器",
    version="1.0.0",
    lifespan=lifespan,
)

# 啟用跨來源資源共享
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    user_id: int
    depth: Union[int, str] = 3


class FidelityProbeRequest(BaseModel):
    user_id: int
    item_id: int
    depth: Union[int, str] = 3
    mode: str = "normal"  # normal, f_plus, f_minus
    path_nodes: Optional[Union[List[int], List[List[int]]]] = None


class ServerState:
    def __init__(self):
        self.device = torch.device("cpu")
        self.data_dir = os.path.join(project_root, "data", "processed")
        self.raw_data_dir = os.path.join(project_root, "data", "raw")
        self.models_dir = os.path.join(project_root, "models")
        self.demo_cases_path = os.path.join(project_root, "src", "demo", "web", "demo_cases.json")

        self.stats = {}
        self.n_users = 0
        self.n_items = 0
        self.n_entities = 0
        self.n_relations = 0
        self.user_le = None
        self.item_le = None
        self.entity_maps = None
        self.recipe_name_map = {}
        self.test_user_items = {}

        self.indices = None
        self.edge_types = None
        self.num_nodes = 0
        self.edge_hash = None
        self.src = None
        self.dst = None

        self.sparse_adj = None
        self.csr_adj = None
        self.bipartite_indices = None
        self.bipartite_num_nodes = 0

        self.loaded_models: Dict[str, Any] = {}
        self.explainers: Dict[str, Any] = {}
        self.demo_cache: Dict[str, Any] = {}
        self.dynamic_user_cache: Dict[str, Any] = {}
        self.is_initialized = False

    def initialize(self):
        if self.is_initialized:
            return

        print("初始化伺服器資源...")
        # 設備選擇
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device = torch.device("xpu")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"推論計算設備: {self.device}")

        # 載入離線預載資料
        if os.path.exists(self.demo_cases_path):
            with open(self.demo_cases_path, "r", encoding="utf-8") as f:
                self.demo_cache = json.load(f)
            print("已載入 demo_cases.json 快取資料")

        # 載入名稱對應表
        self.user_le, self.item_le, self.entity_maps, self.recipe_name_map = fast_load_names_and_maps(
            self.data_dir, self.raw_data_dir
        )

        # 載入統計資訊
        stats_path = os.path.join(self.data_dir, "stats.pkl")
        if os.path.exists(stats_path):
            with open(stats_path, "rb") as f:
                self.stats = pickle.load(f)
            self.n_users = self.stats["n_users"]
            self.n_items = self.stats["n_items"]
            self.n_entities = self.stats["n_entities"]
            self.n_relations = self.stats["n_relations"]

        # 建立測試集映射
        self._build_test_interactions()

        # 載入圖結構
        self._load_graph_structure()

        self.is_initialized = True

        # 預載預設模型 (深度 3)
        self.get_or_load_model("3")
        print("伺服器初始化完成！")

    def _build_test_interactions(self):
        interactions_path = os.path.join(self.data_dir, "interactions.pkl")
        if not os.path.exists(interactions_path):
            return
        with open(interactions_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, pd.DataFrame):
            arr = data[["user_id_remap", "recipe_id_remap"]].values
        else:
            arr = np.array(data)[:, :2]

        np.random.seed(42)
        indices = np.arange(len(arr))
        np.random.shuffle(indices)
        split_idx = int(len(arr) * 0.8)
        test_arr = arr[indices[split_idx:]]

        for row in test_arr:
            u, i = int(row[0]), int(row[1])
            if u not in self.test_user_items:
                self.test_user_items[u] = set()
            self.test_user_items[u].add(i)

    def _load_graph_structure(self):
        kg_path = os.path.join(self.data_dir, "kg_triples.pkl")
        int_path = os.path.join(self.data_dir, "interactions.pkl")
        if not os.path.exists(kg_path) or not os.path.exists(int_path):
            return

        with open(kg_path, "rb") as f:
            kg_triples = pickle.load(f)
        with open(int_path, "rb") as f:
            interactions = pickle.load(f)
        if isinstance(interactions, pd.DataFrame):
            interactions = interactions.values

        indices, edge_types, num_nodes = get_adj_indices(
            kg_triples, interactions, self.n_users, self.n_items, self.n_entities
        )
        self.indices = indices.to(self.device)
        self.edge_types = edge_types.to(self.device)
        self.num_nodes = num_nodes

        self.src = self.indices[0]
        self.dst = self.indices[1]
        self.edge_hash = self.src * self.num_nodes + self.dst

        # 建構全圖 CSR 鄰接索引供向量化快速尋路使用 (耗時約 0.18 秒，記憶體約 50MB)
        src_np = indices[0].numpy()
        dst_np = indices[1].numpy()
        self.csr_adj = sp.csr_matrix(
            (np.arange(len(src_np), dtype=np.int32) + 1, (src_np, dst_np)),
            shape=(self.num_nodes, self.num_nodes),
        )
        self.csr_adj.sort_indices()

        # 提取二分圖指標供 LightGCN 使用
        bipartite_indices, bipartite_nodes = get_user_item_indices(interactions, self.n_users, self.n_items)
        self.bipartite_indices = bipartite_indices.to(self.device)
        self.bipartite_num_nodes = bipartite_nodes

    def get_sparse_adj(self):
        if self.sparse_adj is None:
            kg_path = os.path.join(self.data_dir, "kg_triples.pkl")
            int_path = os.path.join(self.data_dir, "interactions.pkl")
            with open(kg_path, "rb") as f:
                kg_triples = pickle.load(f)
            with open(int_path, "rb") as f:
                interactions = pickle.load(f)
            if isinstance(interactions, pd.DataFrame):
                interactions = interactions.values
            self.sparse_adj = construct_adj(
                kg_triples, interactions, self.n_users, self.n_items, self.n_entities
            ).to(self.device)
        return self.sparse_adj

    def get_or_load_model(self, depth: Union[int, str]):
        if not self.is_initialized:
            self.initialize()
        depth_key = str(depth)
        if depth_key in self.loaded_models:
            return self.loaded_models[depth_key], self.explainers.get(depth_key)

        ckpt_map = {
            "3": os.path.join(self.models_dir, "depth_3", "1_kgat_checkpoint_e3.pth"),
            "2": os.path.join(self.models_dir, "depth_2", "1_kgat_checkpoint_e1.pth"),
            "1": os.path.join(self.models_dir, "full_kgat", "2_kgat_checkpoint_e9.pth"),
            "wo_kg": os.path.join(self.models_dir, "wo_kg", "2_kgat_checkpoint_e10.pth"),
            "wo_attn": os.path.join(self.models_dir, "wo_attn", "3_kgat_checkpoint_e23.pth"),
            "lightgcn": os.path.join(self.models_dir, "baseline", "3_lightgcn_checkpoint_e11.pth"),
            "nfm": os.path.join(self.models_dir, "baseline", "3_nfm_checkpoint_e35.pth"),
            "bprmf": os.path.join(self.models_dir, "baseline", "3_bprmf_checkpoint_e61.pth"),
        }

        ckpt_path = ckpt_map.get(depth_key)
        if not ckpt_path or not os.path.exists(ckpt_path):
            print(f"模型檔案不存在: {ckpt_path}")
            return None, None

        print(f"載入檢查點: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        saved_args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
        del checkpoint
        gc.collect()
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        if depth_key in ("3", "2", "1", "wo_kg"):
            embed_dim = getattr(saved_args, "embed_dim", 64) if saved_args else 64
            layers = getattr(saved_args, "layers", [64, 32]) if saved_args else [64, 32]

            model = KGATAttention(
                self.n_users,
                self.n_items + self.n_entities,
                self.n_relations + 2,
                embed_dim=embed_dim,
                layers=layers,
            )
            model.load_state_dict(state_dict, strict=False)
            if getattr(saved_args, "use_bf16", False):
                model = model.to(dtype=torch.bfloat16)
                for l in model.aggregator_layers:
                    l.force_fp32_agg = False
            model = model.to(self.device)
            model.eval()
            with torch.no_grad():
                model.cached_final_embed = model.get_final_embeddings(
                    self.indices, self.edge_types, self.num_nodes
                )
            explainer = KGATAttentionExplainer(model) if depth_key != "wo_kg" else None
        elif depth_key == "wo_attn":
            embed_dim = getattr(saved_args, "embed_dim", 64) if saved_args else 64
            layers = getattr(saved_args, "layers", [64]) if saved_args else [64]

            model = KGAT_BiInteraction(
                self.n_users,
                self.n_items + self.n_entities,
                self.n_relations,
                embed_dim=embed_dim,
                layers=layers,
            ).to(self.device)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            with torch.no_grad():
                model.cached_final_embed = model.get_final_embeddings(self.get_sparse_adj())
            explainer = None
        elif depth_key == "lightgcn":
            model = LightGCN(self.n_users, self.n_items, embed_dim=64, layers=3).to(self.device)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            with torch.no_grad():
                model.cached_final_embed = model.get_final_embeddings(self.bipartite_indices, self.bipartite_num_nodes)
            explainer = None
        elif depth_key == "bprmf":
            model = BPRMF(self.n_users, self.n_items, embed_dim=64).to(self.device)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            explainer = None
        elif depth_key == "nfm":
            model = NFM(self.n_users, self.n_items, embed_dim=64).to(self.device)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            explainer = None
        else:
            return None, None

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.loaded_models[depth_key] = model
        self.explainers[depth_key] = explainer
        return model, explainer


state = ServerState()


@app.get("/api/status")
def get_status():
    """回傳主機連線狀態與目前載入之模型設定."""
    return {
        "status": "連線",
        "device": str(state.device),
        "port": 8501,
        "n_users": state.n_users,
        "n_items": state.n_items,
        "n_entities": state.n_entities,
        "n_relations": state.n_relations,
        "available_depths": ["3", "2", "1", "wo_kg", "wo_attn", "lightgcn", "nfm", "bprmf"],
        "benchmark_metrics": state.demo_cache.get("benchmark_metrics", {}),
    }


@app.post("/api/recommend")
def recommend(req: RecommendRequest):
    """依使用者編號與傳播深度，回傳推薦結果與注意力解釋路徑."""
    try:
        uid = req.user_id
        depth_key = str(req.depth)

        # 驗證使用者編號有效邊界
        if uid < 0 or uid >= state.n_users:
            raise HTTPException(status_code=400, detail=f"使用者編號超出範圍 (0 ~ {state.n_users - 1})")

        # 驗證模型代碼
        if depth_key not in ("3", "2", "1", "wo_kg", "wo_attn", "lightgcn", "nfm", "bprmf"):
            raise HTTPException(status_code=400, detail="不支援的模型代碼")

        # 1. 優先查閱評估快取與動態使用者快取
        cache_key = f"{uid}_{depth_key}"
        if cache_key in state.dynamic_user_cache:
            logger.info(f"命中動態使用者快取: 使用者 {uid}, 深度 {depth_key}")
            return state.dynamic_user_cache[cache_key]

        cached_user = state.demo_cache.get("user_cache", {}).get(str(uid))
        if cached_user and depth_key in cached_user.get("depths", {}):
            cached_depth_data = cached_user["depths"][depth_key]
            if cached_depth_data is not None:
                metrics = state.demo_cache.get("benchmark_metrics", {}).get(depth_key, {})
                logger.info(f"命中快取推薦: 使用者 {uid}, 深度 {depth_key}")
                return {
                    "user_id": uid,
                    "user_name": cached_user.get("user_name", f"使用者 {uid}"),
                    "depth": depth_key,
                    "recommended_item_id": cached_depth_data["recommended_item_id"],
                    "recommended_item_name": cached_depth_data["recommended_item_name"],
                    "score": cached_depth_data["score"],
                    "prob": cached_depth_data["prob"],
                    "is_test_hit": cached_depth_data["is_test_hit"],
                    "user_n_interactions": cached_depth_data["user_n_interactions"],
                    "item_kg_degree": cached_depth_data["item_kg_degree"],
                    "metrics": {
                        "hr_20": metrics.get("hr_20", 0.0),
                        "ndcg_20": metrics.get("ndcg_20", 0.0),
                    },
                    "explanations": cached_depth_data["explanations"],
                    "fidelity": cached_depth_data.get("fidelity", {}),
                }

        # 2. 動態推論 (以快取 Embeddings 向量內積計算，消除顯存壓力)
        model, explainer = state.get_or_load_model(depth_key)
        if model is None:
            raise HTTPException(status_code=500, detail="模型未載入")

        if depth_key in ("3", "2", "1", "wo_kg", "wo_attn", "lightgcn"):
            with torch.no_grad():
                u_embed = model.cached_final_embed[uid]
                items_embed = model.cached_final_embed[state.n_users : state.n_users + state.n_items]
                pos_scores = torch.matmul(items_embed, u_embed).float()
                best_item_idx = int(torch.argmax(pos_scores).item())
                best_score = float(pos_scores[best_item_idx].item())
                prob = float(torch.sigmoid(torch.tensor(best_score)).item())
        elif depth_key == "bprmf":
            with torch.no_grad():
                u_embed = model.user_embed.weight[uid]
                items_embed = model.item_embed.weight
                pos_scores = torch.matmul(items_embed, u_embed).float()
                best_item_idx = int(torch.argmax(pos_scores).item())
                best_score = float(pos_scores[best_item_idx].item())
                prob = float(torch.sigmoid(torch.tensor(best_score)).item())
        elif depth_key == "nfm":
            with torch.no_grad():
                u_emb = model.user_embed.weight[uid]
                u_b = model.user_bias.weight[uid]
                bi = u_emb * model.item_embed.weight
                mlp = model.mlp(bi).squeeze(-1)
                pos_scores = u_b + model.item_bias.weight.squeeze(-1) + model.global_bias + mlp
                best_item_idx = int(torch.argmax(pos_scores).item())
                best_score = float(pos_scores[best_item_idx].item())
                prob = float(torch.sigmoid(torch.tensor(best_score)).item())
        else:
            raise HTTPException(status_code=400, detail="不支援的模型代碼")

        # 取得節點名稱
        _, rec_real_id, rec_name = get_node_name(
            state.n_users + best_item_idx,
            state.n_users,
            state.n_items,
            state.user_le,
            state.item_le,
            state.entity_maps,
            state.recipe_name_map,
        )
        _, _, user_name = get_node_name(
            uid,
            state.n_users,
            state.n_items,
            state.user_le,
            state.item_le,
            state.entity_maps,
            state.recipe_name_map,
        )

        # 取得注意力解釋路徑 (優先使用 CSR 向量化尋路，耗時數毫秒)
        n_hops = 1 if depth_key == "1" else (2 if depth_key == "2" else 2)
        explanations = []
        if explainer and depth_key in ("3", "2", "1"):
            if state.csr_adj is not None:
                exp_res = explainer.explain_fast_csr(
                    state.indices, state.edge_types, state.num_nodes, uid, best_item_idx, state.csr_adj, top_k=3, n_hops=n_hops
                )
            else:
                exp_res = explainer.explain(
                    state.indices, state.edge_types, state.num_nodes, uid, best_item_idx, n_hops=n_hops, top_k=3
                )
            if exp_res and "top_paths" in exp_res:
                for path, score in exp_res["top_paths"]:
                    path_details = []
                    path_desc_list = []
                    for node_id in path:
                        node_id = int(node_id)
                        n_type, n_real_id, n_name = get_node_name(
                            node_id,
                            state.n_users,
                            state.n_items,
                            state.user_le,
                            state.item_le,
                            state.entity_maps,
                            state.recipe_name_map,
                        )
                        path_details.append({
                            "id_remapped": node_id,
                            "id_original": n_real_id,
                            "type": n_type,
                            "name": n_name,
                        })
                        path_desc_list.append(n_name)
                    explanations.append({
                        "path_structure": " -> ".join([p["type"] for p in path_details]),
                        "path_description": " -> ".join(path_desc_list),
                        "path_details": path_details,
                        "contribution_score": float(score),
                    })

        is_hit = best_item_idx in state.test_user_items.get(uid, set())
        metrics = state.demo_cache.get("benchmark_metrics", {}).get(depth_key, {})

        logger.info(f"推薦成功: 使用者 {uid}, 深度 {depth_key}, 推薦食譜 {best_item_idx} ({rec_name})")
        res_data = {
            "user_id": uid,
            "user_name": user_name,
            "depth": depth_key,
            "recommended_item_id": best_item_idx,
            "recommended_item_name": rec_name,
            "score": best_score,
            "prob": prob,
            "is_test_hit": is_hit,
            "user_n_interactions": len(state.test_user_items.get(uid, set())),
            "item_kg_degree": 0,
            "metrics": {
                "hr_20": metrics.get("hr_20", 0.0),
                "ndcg_20": metrics.get("ndcg_20", 0.0),
            },
            "explanations": explanations,
            "fidelity": {},
        }
        state.dynamic_user_cache[cache_key] = res_data
        return res_data
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"推薦端點執行異常: 使用者 {req.user_id}, 深度 {req.depth}: {exc}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
        raise HTTPException(status_code=500, detail=f"推論伺服器執行異常: {str(exc)}")


@app.post("/api/fidelity_probe")
def fidelity_probe(req: FidelityProbeRequest):
    """忠實度結構探針端點，支援 normal, f_plus, f_minus 三態圖遮蔽計算."""
    try:
        uid = req.user_id
        item_id = req.item_id
        depth_key = str(req.depth)
        mode = req.mode.lower()

        if mode not in ("normal", "f_plus", "f_minus"):
            raise HTTPException(status_code=400, detail="無效探針模式，支援 normal, f_plus, f_minus")

        # 驗證使用者編號與食譜編號邊界
        if uid < 0 or uid >= state.n_users:
            raise HTTPException(status_code=400, detail=f"使用者編號超出範圍 (0 ~ {state.n_users - 1})")
        if item_id < 0 or item_id >= state.n_items:
            raise HTTPException(status_code=400, detail=f"食譜編號超出範圍 (0 ~ {state.n_items - 1})")

        # 正規化 path_nodes 支援單一路徑 (List[int]) 或多路徑聯集 (List[List[int]])
        normalized_paths = None
        if req.path_nodes is not None:
            if len(req.path_nodes) > 0 and isinstance(req.path_nodes[0], (int, np.integer)):
                normalized_paths = [req.path_nodes]
            else:
                normalized_paths = req.path_nodes

        # 查核快取 (含靜態評估快取與動態推論快取)
        cache_key = f"{uid}_{depth_key}"
        cached_user = state.demo_cache.get("user_cache", {}).get(str(uid))
        cached_depth = cached_user.get("depths", {}).get(depth_key) if cached_user else None
        if not cached_depth and cache_key in state.dynamic_user_cache:
            cached_depth = state.dynamic_user_cache[cache_key]

        # 針對非圖注意力模型或無路徑使用者，優雅回傳零變化指標，絕不拋出 500
        if depth_key not in ("3", "2", "1") or (normalized_paths is not None and not any(len(p) >= 2 for p in normalized_paths)):
            if cached_depth and "prob" in cached_depth:
                p_orig = cached_depth["prob"]
            else:
                model, _ = state.get_or_load_model(depth_key)
                if model is not None and hasattr(model, "cached_final_embed"):
                    with torch.no_grad():
                        u_embed = model.cached_final_embed[uid]
                        item_embed = model.cached_final_embed[state.n_users + item_id]
                        score_orig = float(torch.sum(u_embed.float() * item_embed.float()).item())
                        p_orig = float(torch.sigmoid(torch.tensor(score_orig)).item())
                elif model is not None and depth_key in ("3", "2", "1", "wo_kg") and state.indices is not None:
                    u_tensor = torch.tensor([uid], dtype=torch.long, device=state.device)
                    i_tensor = torch.tensor([item_id], dtype=torch.long, device=state.device)
                    with torch.no_grad():
                        score_orig = model(state.indices, state.edge_types, state.num_nodes, u_tensor, i_tensor)[0].item()
                        p_orig = float(torch.sigmoid(torch.tensor(score_orig)).item())
                elif model is not None and depth_key == "bprmf":
                    with torch.no_grad():
                        u_embed = model.user_embed.weight[uid]
                        item_embed = model.item_embed.weight[item_id]
                        score_orig = float(torch.sum(u_embed.float() * item_embed.float()).item())
                        p_orig = float(torch.sigmoid(torch.tensor(score_orig)).item())
                elif model is not None and depth_key == "nfm":
                    with torch.no_grad():
                        u_emb = model.user_embed.weight[uid]
                        i_emb = model.item_embed.weight[item_id]
                        u_b = model.user_bias.weight[uid]
                        i_b = model.item_bias.weight[item_id]
                        bi = (u_emb * i_emb).unsqueeze(0)
                        mlp = model.mlp(bi).squeeze()
                        score_orig = float((u_b + i_b + model.global_bias + mlp).item())
                        p_orig = float(torch.sigmoid(torch.tensor(score_orig)).item())
                else:
                    p_orig = 0.0
            logger.info(f"無路徑或基準模型探針回傳: 使用者 {uid}, 食譜 {item_id}, 深度 {depth_key}")
            return {
                "user_id": uid,
                "item_id": item_id,
                "depth": depth_key,
                "mode": mode,
                "prob": p_orig,
                "original_prob": p_orig,
                "fidelity_score": 0.0,
                "status_description": "無路徑使用者",
                "is_no_path": True,
                "note": "無路徑使用者",
                "f_plus": 0.0,
                "f_minus": 0.0,
            }

        # 如果有快取且為同一推薦物品（且未指定自訂 path_nodes）
        if not normalized_paths and cached_depth and cached_depth.get("recommended_item_id") == item_id:
            p_orig = cached_depth["prob"]
            fid = cached_depth.get("fidelity", {})
            if mode == "normal":
                return {
                    "user_id": uid,
                    "item_id": item_id,
                    "depth": depth_key,
                    "mode": "normal",
                    "prob": p_orig,
                    "original_prob": p_orig,
                    "fidelity_score": 0.0,
                    "status_description": "原始預測狀態",
                }
            elif mode == "f_plus":
                p_f_plus = fid.get("prob_f_plus")
                f_plus_val = fid.get("f_plus")
                if p_f_plus is not None and f_plus_val is not None:
                    return {
                        "user_id": uid,
                        "item_id": item_id,
                        "depth": depth_key,
                        "mode": "f_plus",
                        "prob": p_f_plus,
                        "original_prob": p_orig,
                        "fidelity_score": f_plus_val,
                        "status_description": "遮蔽解釋子圖：驗證必要性",
                    }
            elif mode == "f_minus":
                p_f_minus = fid.get("prob_f_minus")
                f_minus_val = fid.get("f_minus")
                if p_f_minus is not None and f_minus_val is not None:
                    return {
                        "user_id": uid,
                        "item_id": item_id,
                        "depth": depth_key,
                        "mode": "f_minus",
                        "prob": p_f_minus,
                        "original_prob": p_orig,
                        "fidelity_score": f_minus_val,
                        "status_description": "僅保留解釋子圖：驗證充分性",
                    }

        # 動態圖遮蔽計算
        model, explainer = state.get_or_load_model(depth_key)
        if model is None or state.indices is None:
            raise HTTPException(status_code=500, detail="模型或圖結構未載入")

        u_tensor = torch.tensor([uid], dtype=torch.long, device=state.device)
        i_tensor = torch.tensor([item_id], dtype=torch.long, device=state.device)

        # 1. 計算原始預測 (優先採用特徵快取)
        with torch.no_grad():
            if hasattr(model, "cached_final_embed"):
                u_embed = model.cached_final_embed[uid]
                item_embed = model.cached_final_embed[state.n_users + item_id]
                score_orig = float(torch.sum(u_embed.float() * item_embed.float()).item())
                p_orig = float(torch.sigmoid(torch.tensor(score_orig)).item())
            else:
                score_orig = model(state.indices, state.edge_types, state.num_nodes, u_tensor, i_tensor)[0].item()
                p_orig = float(torch.sigmoid(torch.tensor(score_orig)).item())

        if mode == "normal":
            logger.info(f"原始探針查詢: 使用者 {uid}, 食譜 {item_id}, 機率 {p_orig:.4f}")
            return {
                "user_id": uid,
                "item_id": item_id,
                "depth": depth_key,
                "mode": "normal",
                "prob": p_orig,
                "original_prob": p_orig,
                "fidelity_score": 0.0,
                "status_description": "原始預測狀態",
            }

        # 2. 獲取解釋路徑邊緣
        important_edges = set()
        if normalized_paths:
            for path in normalized_paths:
                for step in range(len(path) - 1):
                    n1, n2 = int(path[step]), int(path[step + 1])
                    important_edges.add((n1, n2))
                    important_edges.add((n2, n1))
        elif cached_depth and cached_depth.get("explanations"):
            for exp in cached_depth["explanations"]:
                details = exp.get("path_details", [])
                for step in range(len(details) - 1):
                    n1, n2 = int(details[step]["id_remapped"]), int(details[step + 1]["id_remapped"])
                    important_edges.add((n1, n2))
                    important_edges.add((n2, n1))
        elif explainer:
            n_hops = 1 if depth_key == "1" else 2
            if state.csr_adj is not None:
                exp_res = explainer.explain_fast_csr(state.indices, state.edge_types, state.num_nodes, uid, item_id, state.csr_adj, top_k=3, n_hops=n_hops)
            else:
                exp_res = explainer.explain(state.indices, state.edge_types, state.num_nodes, uid, item_id, n_hops=n_hops, top_k=3)
            if exp_res and "top_paths" in exp_res:
                for path, _ in exp_res["top_paths"]:
                    for step in range(len(path) - 1):
                        n1, n2 = int(path[step]), int(path[step + 1])
                        important_edges.add((n1, n2))
                        important_edges.add((n2, n1))

        if not important_edges:
            # 若無重要邊緣可遮蔽，返回原值
            logger.info(f"無可遮蔽邊緣: 使用者 {uid}, 食譜 {item_id}")
            return {
                "user_id": uid,
                "item_id": item_id,
                "depth": depth_key,
                "mode": mode,
                "prob": p_orig,
                "original_prob": p_orig,
                "fidelity_score": 0.0,
                "status_description": "無路徑使用者",
                "is_no_path": True,
                "note": "無路徑使用者",
                "f_plus": 0.0,
                "f_minus": 0.0,
            }

        hashed_important = [n1 * state.num_nodes + n2 for n1, n2 in important_edges]
        hashed_tensor = torch.tensor(hashed_important, device=state.device)
        mask_important = torch.isin(state.edge_hash, hashed_tensor)
        mask_self = state.src == state.dst

        if mode == "f_plus":
            # 移除重要路徑
            mask_f_plus = ~mask_important
            with torch.no_grad():
                f_plus_score = model(
                    state.indices[:, mask_f_plus],
                    state.edge_types[mask_f_plus],
                    state.num_nodes,
                    u_tensor,
                    i_tensor,
                )[0].item()
                p_f_plus = float(torch.sigmoid(torch.tensor(f_plus_score)).item())
                f_plus_val = p_orig - p_f_plus
            logger.info(f"F+ 遮蔽計算成功: 使用者 {uid}, 食譜 {item_id}, F+={f_plus_val:.4f}")
            res_data = {
                "user_id": uid,
                "item_id": item_id,
                "depth": depth_key,
                "mode": "f_plus",
                "prob": p_f_plus,
                "original_prob": p_orig,
                "fidelity_score": f_plus_val,
                "status_description": "遮蔽解釋子圖：驗證必要性",
            }
            if cache_key in state.dynamic_user_cache:
                if "fidelity" not in state.dynamic_user_cache[cache_key]:
                    state.dynamic_user_cache[cache_key]["fidelity"] = {}
                state.dynamic_user_cache[cache_key]["fidelity"]["f_plus"] = f_plus_val
                state.dynamic_user_cache[cache_key]["fidelity"]["prob_f_plus"] = p_f_plus
            return res_data
        else:  # f_minus
            # 僅保留重要路徑與自迴圈
            mask_f_minus = mask_important | mask_self
            with torch.no_grad():
                f_minus_score = model(
                    state.indices[:, mask_f_minus],
                    state.edge_types[mask_f_minus],
                    state.num_nodes,
                    u_tensor,
                    i_tensor,
                )[0].item()
                p_f_minus = float(torch.sigmoid(torch.tensor(f_minus_score)).item())
                f_minus_val = p_orig - p_f_minus
            logger.info(f"F- 遮蔽計算成功: 使用者 {uid}, 食譜 {item_id}, F-={f_minus_val:.4f}")
            res_data = {
                "user_id": uid,
                "item_id": item_id,
                "depth": depth_key,
                "mode": "f_minus",
                "prob": p_f_minus,
                "original_prob": p_orig,
                "fidelity_score": f_minus_val,
                "status_description": "僅保留解釋子圖：驗證充分性",
            }
            if cache_key in state.dynamic_user_cache:
                if "fidelity" not in state.dynamic_user_cache[cache_key]:
                    state.dynamic_user_cache[cache_key]["fidelity"] = {}
                state.dynamic_user_cache[cache_key]["fidelity"]["f_minus"] = f_minus_val
                state.dynamic_user_cache[cache_key]["fidelity"]["prob_f_minus"] = p_f_minus
            return res_data
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"探針端點執行異常: 使用者 {req.user_id}, 食譜 {req.item_id}, 模式 {req.mode}: {exc}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
        raise HTTPException(status_code=500, detail=f"探針運算異常: {str(exc)}")


# 掛載前端網頁靜態資源
web_dir = os.path.join(project_root, "src", "demo", "web")
if os.path.exists(web_dir):
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="static")


def run():
    parser = argparse.ArgumentParser(description="KGAT 展示推論伺服器")
    parser.add_argument("--port", type=int, default=8501, help="伺服器監聽端口，預設為 8501")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="伺服器監聽位址")
    args = parser.parse_args()

    if args.port == 8000:
        print("錯誤：為防止連接埠衝突，嚴禁使用 8000 端口，自動重設為 8501")
        args.port = 8501

    print(f"啟動展示推論伺服器：http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    run()
