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
from typing import Any, Dict, List, Optional, Set, Tuple, Union

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

current_dir = Path(__file__).resolve().parent
project_root = current_dir if (current_dir / "data").exists() else current_dir.parent
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


MODEL_CHECKPOINT_RELPATHS: Dict[str, Tuple[str, ...]] = {
    "3": ("depth_3", "1_kgat_checkpoint_e3.pth"),
    "2": ("depth_2", "1_kgat_checkpoint_e1.pth"),
    "1": ("full_kgat", "2_kgat_checkpoint_e9.pth"),
    "wo_kg": ("wo_kg", "2_kgat_checkpoint_e10.pth"),
    "wo_attn": ("wo_attn", "3_kgat_checkpoint_e23.pth"),
    "lightgcn": ("baseline", "3_lightgcn_checkpoint_e11.pth"),
    "nfm": ("baseline", "3_nfm_checkpoint_e35.pth"),
    "bprmf": ("baseline", "3_bprmf_checkpoint_e61.pth"),
}


def _clear_accelerator_cache() -> None:
    """清空可用的 XPU 或 CUDA 裝置記憶體快取."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


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

DEFAULT_PORT = 8501
FORBIDDEN_PORT = 8000
DEFAULT_CORS_ORIGIN_REGEX = (
    r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$"
    r"|^https://([a-zA-Z0-9-]+\.)?(github\.io|vercel\.app|trycloudflare\.com)$"
)


def get_default_port() -> int:
    """讀取環境變數 PORT，無效值或禁占 8000 時回退至 8501."""
    raw = os.environ.get("PORT", str(DEFAULT_PORT)).strip()
    try:
        port = int(raw)
    except ValueError:
        return DEFAULT_PORT
    return DEFAULT_PORT if port == FORBIDDEN_PORT or port <= 0 or port > 65535 else port


def build_cors_config() -> Dict[str, Any]:
    """建立 CORS 設定；停用憑證傳輸，支援 CORS_ORIGINS 環境變數顯式覆蓋."""
    raw_origins = os.environ.get("CORS_ORIGINS", "").strip()
    config: Dict[str, Any] = {
        "allow_credentials": False,
        "allow_methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["*"],
    }
    if raw_origins:
        config["allow_origins"] = [o.strip() for o in raw_origins.split(",") if o.strip()]
    else:
        config["allow_origins"] = []
        config["allow_origin_regex"] = DEFAULT_CORS_ORIGIN_REGEX
    return config


# 啟用跨來源資源共享
app.add_middleware(CORSMiddleware, **build_cors_config())


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
        self.port = get_default_port()
        self.data_dir = os.path.join(project_root, "data", "processed")
        self.raw_data_dir = os.path.join(project_root, "data", "raw")
        self.models_dir = os.path.join(project_root, "models")
        self.demo_cases_path = (
            os.path.join(current_dir, "web", "demo_cases.json")
            if os.path.exists(os.path.join(current_dir, "web", "demo_cases.json"))
            else os.path.join(project_root, "demo", "web", "demo_cases.json")
        )

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
        self.user_interaction_counts = None

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

        # 預計算每位使用者之全圖歷史互動總數
        self.user_interaction_counts = np.zeros(self.n_users, dtype=np.int32)
        for row in arr:
            u = int(row[0])
            if u < self.n_users:
                self.user_interaction_counts[u] += 1

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

    def count_equivalent_channels(self, uid: int, item_idx: int) -> int:
        """計算使用者與目標食譜之間的所有等價 3-hop 通道數量 (u -> r -> mid -> target)."""
        if self.csr_adj is None or uid >= self.num_nodes or (self.n_users + item_idx) >= self.num_nodes:
            return 0
        target_node = self.n_users + item_idx
        u_neighbors = self.csr_adj[uid].indices
        target_neighbors = set(self.csr_adj[target_node].indices)
        total = 0
        for r in u_neighbors:
            r_neighbors = set(self.csr_adj[r].indices)
            common_mid = (r_neighbors & target_neighbors) - {uid, target_node}
            total += len(common_mid)
        return total

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

    def _build_model_and_explainer(self, depth_key: str, state_dict: dict, saved_args: Any):
        """依模型代碼實例化對應架構、載入權重並預計算節點嵌入快取."""
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
                for layer in model.aggregator_layers:
                    layer.force_fp32_agg = False
            model = model.to(self.device)
            model.eval()
            with torch.no_grad():
                model.cached_final_embed = model.get_final_embeddings(
                    self.indices, self.edge_types, self.num_nodes
                )
            explainer = KGATAttentionExplainer(model) if depth_key != "wo_kg" else None
            return model, explainer

        if depth_key == "wo_attn":
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
            return model, None

        if depth_key == "lightgcn":
            model = LightGCN(self.n_users, self.n_items, embed_dim=64, layers=3).to(self.device)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            with torch.no_grad():
                model.cached_final_embed = model.get_final_embeddings(
                    self.bipartite_indices, self.bipartite_num_nodes
                )
            return model, None

        if depth_key in ("bprmf", "nfm"):
            cls = BPRMF if depth_key == "bprmf" else NFM
            model = cls(self.n_users, self.n_items, embed_dim=64).to(self.device)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            return model, None

        return None, None

    def get_or_load_model(self, depth: Union[int, str]):
        if not self.is_initialized:
            self.initialize()
        depth_key = str(depth)
        if depth_key in self.loaded_models:
            return self.loaded_models[depth_key], self.explainers.get(depth_key)

        rel_parts = MODEL_CHECKPOINT_RELPATHS.get(depth_key)
        ckpt_path = os.path.join(self.models_dir, *rel_parts) if rel_parts else None
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
        _clear_accelerator_cache()

        model, explainer = self._build_model_and_explainer(depth_key, state_dict, saved_args)
        if model is None:
            return None, None

        _clear_accelerator_cache()
        self.loaded_models[depth_key] = model
        self.explainers[depth_key] = explainer
        return model, explainer


state = ServerState()


def _lookup_cached_recommendation(uid: int, depth_key: str) -> Optional[Dict[str, Any]]:
    """查詢動態與靜態快取中的推薦結果，命中時回傳標準化回應字典."""
    cache_key = f"{uid}_{depth_key}"
    if cache_key in state.dynamic_user_cache:
        logger.info(f"命中動態使用者快取: 使用者 {uid}, 深度 {depth_key}")
        return state.dynamic_user_cache[cache_key]

    cached_user = state.demo_cache.get("user_cache", {}).get(str(uid))
    if not cached_user or depth_key not in cached_user.get("depths", {}):
        return None

    cached_depth_data = cached_user["depths"][depth_key]
    if cached_depth_data is None:
        return None

    metrics = state.demo_cache.get("benchmark_metrics", {}).get(depth_key, {})
    logger.info(f"命中快取推薦: 使用者 {uid}, 深度 {depth_key}")
    rec_item = cached_depth_data["recommended_item_id"]
    user_n_interactions = (
        int(state.user_interaction_counts[uid])
        if state.user_interaction_counts is not None and uid < len(state.user_interaction_counts)
        else cached_depth_data.get("user_n_interactions", 0)
    )
    n_equivalent_channels = cached_depth_data.get("n_equivalent_channels")
    if not n_equivalent_channels:
        n_equivalent_channels = state.count_equivalent_channels(uid, rec_item)

    return {
        "user_id": uid,
        "user_name": cached_user.get("user_name", f"使用者 {uid}"),
        "depth": depth_key,
        "recommended_item_id": rec_item,
        "recommended_item_name": cached_depth_data["recommended_item_name"],
        "score": cached_depth_data["score"],
        "prob": cached_depth_data["prob"],
        "is_test_hit": cached_depth_data["is_test_hit"],
        "user_n_interactions": user_n_interactions,
        "n_equivalent_channels": n_equivalent_channels,
        "item_kg_degree": cached_depth_data["item_kg_degree"],
        "metrics": {
            "hr_20": metrics.get("hr_20", 0.0),
            "ndcg_20": metrics.get("ndcg_20", 0.0),
        },
        "explanations": cached_depth_data["explanations"],
        "fidelity": cached_depth_data.get("fidelity", {}),
    }


def _score_top_item(model, depth_key: str, uid: int) -> Tuple[int, float, float]:
    """對全體候選食譜計算分數並回傳最高分之 (item_idx, score, prob)."""
    with torch.no_grad():
        if depth_key in ("3", "2", "1", "wo_kg", "wo_attn", "lightgcn"):
            u_embed = model.cached_final_embed[uid]
            items_embed = model.cached_final_embed[state.n_users : state.n_users + state.n_items]
            pos_scores = torch.matmul(items_embed, u_embed).float()
        elif depth_key == "bprmf":
            u_embed = model.user_embed.weight[uid]
            pos_scores = torch.matmul(model.item_embed.weight, u_embed).float()
        elif depth_key == "nfm":
            u_emb = model.user_embed.weight[uid]
            u_b = model.user_bias.weight[uid]
            bi = u_emb * model.item_embed.weight
            mlp = model.mlp(bi).squeeze(-1)
            pos_scores = u_b + model.item_bias.weight.squeeze(-1) + model.global_bias + mlp
        else:
            raise HTTPException(status_code=400, detail="不支援的模型代碼")

        best_item_idx = int(torch.argmax(pos_scores).item())
        best_score = float(pos_scores[best_item_idx].item())
        prob = float(torch.sigmoid(torch.tensor(best_score)).item())
    return best_item_idx, best_score, prob


def _normalize_path_nodes(
    path_nodes: Optional[Union[List[int], List[List[int]]]],
) -> Optional[List[List[int]]]:
    """將單一路徑 (1D) 或多路徑聯集 (2D) 正規化為二維整數串列."""
    if path_nodes is None:
        return None
    if len(path_nodes) > 0 and isinstance(path_nodes[0], (int, np.integer)):
        return [list(path_nodes)]  # type: ignore[arg-type]
    return [list(p) for p in path_nodes]  # type: ignore[arg-type]


def _build_no_path_probe_response(
    uid: int, item_id: int, depth_key: str, mode: str, p_orig: float
) -> Dict[str, Any]:
    """建構無路徑或非注意力模型的安全零變化探針回應."""
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


def _compute_pair_prob(model, depth_key: str, uid: int, item_id: int) -> float:
    """計算指定 (uid, item_id) 的原始預測機率."""
    if model is None:
        return 0.0
    with torch.no_grad():
        if hasattr(model, "cached_final_embed"):
            u_embed = model.cached_final_embed[uid]
            item_embed = model.cached_final_embed[state.n_users + item_id]
            score_orig = float(torch.sum(u_embed.float() * item_embed.float()).item())
        elif depth_key in ("3", "2", "1", "wo_kg") and state.indices is not None:
            u_tensor = torch.tensor([uid], dtype=torch.long, device=state.device)
            i_tensor = torch.tensor([item_id], dtype=torch.long, device=state.device)
            score_orig = model(state.indices, state.edge_types, state.num_nodes, u_tensor, i_tensor)[0].item()
        elif depth_key == "bprmf":
            u_embed = model.user_embed.weight[uid]
            item_embed = model.item_embed.weight[item_id]
            score_orig = float(torch.sum(u_embed.float() * item_embed.float()).item())
        elif depth_key == "nfm":
            u_emb = model.user_embed.weight[uid]
            i_emb = model.item_embed.weight[item_id]
            u_b = model.user_bias.weight[uid]
            i_b = model.item_bias.weight[item_id]
            bi = (u_emb * i_emb).unsqueeze(0)
            mlp = model.mlp(bi).squeeze()
            score_orig = float((u_b + i_b + model.global_bias + mlp).item())
        else:
            return 0.0
        return float(torch.sigmoid(torch.tensor(score_orig)).item())


def _lookup_cached_probe_response(
    uid: int, item_id: int, depth_key: str, mode: str, cached_depth: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """自快取中直接組裝 normal / f_plus / f_minus 探針結果；缺漏時回傳 None."""
    p_orig = cached_depth["prob"]
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

    fid = cached_depth.get("fidelity", {})
    p_masked = fid.get(f"prob_{mode}")
    fid_val = fid.get(mode)
    if p_masked is None or fid_val is None:
        return None

    desc = "遮蔽解釋子圖：驗證必要性" if mode == "f_plus" else "僅保留解釋子圖：驗證充分性"
    return {
        "user_id": uid,
        "item_id": item_id,
        "depth": depth_key,
        "mode": mode,
        "prob": p_masked,
        "original_prob": p_orig,
        "fidelity_score": fid_val,
        "status_description": desc,
    }


def _collect_important_edges(
    uid: int,
    item_id: int,
    depth_key: str,
    normalized_paths: Optional[List[List[int]]],
    cached_depth: Optional[Dict[str, Any]],
    explainer,
) -> Set[Tuple[int, int]]:
    """依自訂路徑、快取解釋或即時尋路收集雙向重要邊緣集合."""
    important_edges: Set[Tuple[int, int]] = set()

    def _add_path_edges(nodes_seq: List[int]) -> None:
        for step in range(len(nodes_seq) - 1):
            n1, n2 = int(nodes_seq[step]), int(nodes_seq[step + 1])
            important_edges.add((n1, n2))
            important_edges.add((n2, n1))

    if normalized_paths:
        for path in normalized_paths:
            _add_path_edges(path)
    elif cached_depth and cached_depth.get("explanations"):
        for exp in cached_depth["explanations"]:
            details = exp.get("path_details", [])
            _add_path_edges([int(d["id_remapped"]) for d in details])
    elif explainer:
        n_hops = 1 if depth_key == "1" else 2
        if state.csr_adj is not None:
            exp_res = explainer.explain_fast_csr(
                state.indices, state.edge_types, state.num_nodes, uid, item_id, state.csr_adj, top_k=3, n_hops=n_hops
            )
        else:
            exp_res = explainer.explain(
                state.indices, state.edge_types, state.num_nodes, uid, item_id, n_hops=n_hops, top_k=3
            )
        if exp_res and "top_paths" in exp_res:
            for path, _ in exp_res["top_paths"]:
                _add_path_edges([int(n) for n in path])

    return important_edges


def _evaluate_masked_subgraph(
    model,
    uid: int,
    item_id: int,
    depth_key: str,
    mode: str,
    p_orig: float,
    mask_important,
    mask_self,
    cache_key: str,
) -> Dict[str, Any]:
    """依 f_plus (遮蔽) 或 f_minus (僅保留) 執行圖遮蔽前向計算並同步動態快取."""
    u_tensor = torch.tensor([uid], dtype=torch.long, device=state.device)
    i_tensor = torch.tensor([item_id], dtype=torch.long, device=state.device)
    edge_mask = (~mask_important) if mode == "f_plus" else (mask_important | mask_self)

    with torch.no_grad():
        masked_score = model(
            state.indices[:, edge_mask],
            state.edge_types[edge_mask],
            state.num_nodes,
            u_tensor,
            i_tensor,
        )[0].item()
        p_masked = float(torch.sigmoid(torch.tensor(masked_score)).item())
        fid_val = p_orig - p_masked

    tag, desc = (
        ("F+", "遮蔽解釋子圖：驗證必要性")
        if mode == "f_plus"
        else ("F-", "僅保留解釋子圖：驗證充分性")
    )
    logger.info(f"{tag} 遮蔽計算成功: 使用者 {uid}, 食譜 {item_id}, {tag}={fid_val:.4f}")

    if cache_key in state.dynamic_user_cache:
        state.dynamic_user_cache[cache_key].setdefault("fidelity", {})
        state.dynamic_user_cache[cache_key]["fidelity"][mode] = fid_val
        state.dynamic_user_cache[cache_key]["fidelity"][f"prob_{mode}"] = p_masked

    return {
        "user_id": uid,
        "item_id": item_id,
        "depth": depth_key,
        "mode": mode,
        "prob": p_masked,
        "original_prob": p_orig,
        "fidelity_score": fid_val,
        "status_description": desc,
    }


@app.get("/api/status")
def get_status():
    """回傳主機連線狀態與目前載入之模型設定."""
    return {
        "status": "連線",
        "device": str(state.device),
        "port": state.port,
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

        if uid < 0 or uid >= state.n_users:
            raise HTTPException(status_code=400, detail=f"使用者編號超出範圍 (0 ~ {state.n_users - 1})")
        if depth_key not in MODEL_CHECKPOINT_RELPATHS:
            raise HTTPException(status_code=400, detail="不支援的模型代碼")

        cached_res = _lookup_cached_recommendation(uid, depth_key)
        if cached_res is not None:
            return cached_res

        model, explainer = state.get_or_load_model(depth_key)
        if model is None:
            raise HTTPException(status_code=500, detail="模型未載入")

        best_item_idx, best_score, prob = _score_top_item(model, depth_key, uid)
        _, _, rec_name = get_node_name(
            state.n_users + best_item_idx,
            state.n_users,
            state.n_items,
            state.user_le,
            state.item_le,
            state.entity_maps,
            state.recipe_name_map,
        )
        user_name = f"User {uid}"

        # 取得注意力解釋路徑 (優先使用 CSR 向量化尋路，耗時數毫秒)
        n_hops = 1 if depth_key == "1" else 2
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
                        if n_type == "USER":
                            n_name = f"User {node_id}"
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
        n_equivalent_channels = state.count_equivalent_channels(uid, best_item_idx)
        user_n_interactions = (
            int(state.user_interaction_counts[uid])
            if state.user_interaction_counts is not None and uid < len(state.user_interaction_counts)
            else len(state.test_user_items.get(uid, set()))
        )

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
            "user_n_interactions": user_n_interactions,
            "n_equivalent_channels": n_equivalent_channels,
            "item_kg_degree": 0,
            "metrics": {
                "hr_20": metrics.get("hr_20", 0.0),
                "ndcg_20": metrics.get("ndcg_20", 0.0),
            },
            "explanations": explanations,
            "fidelity": {},
        }
        state.dynamic_user_cache[f"{uid}_{depth_key}"] = res_data
        return res_data
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"推薦端點執行異常: 使用者 {req.user_id}, 深度 {req.depth}: {exc}")
        _clear_accelerator_cache()
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
        if uid < 0 or uid >= state.n_users:
            raise HTTPException(status_code=400, detail=f"使用者編號超出範圍 (0 ~ {state.n_users - 1})")
        if item_id < 0 or item_id >= state.n_items:
            raise HTTPException(status_code=400, detail=f"食譜編號超出範圍 (0 ~ {state.n_items - 1})")

        normalized_paths = _normalize_path_nodes(req.path_nodes)

        cache_key = f"{uid}_{depth_key}"
        cached_user = state.demo_cache.get("user_cache", {}).get(str(uid))
        cached_depth = cached_user.get("depths", {}).get(depth_key) if cached_user else None
        if not cached_depth and cache_key in state.dynamic_user_cache:
            cached_depth = state.dynamic_user_cache[cache_key]

        # 針對非圖注意力模型或無路徑使用者，優雅回傳零變化指標，絕不拋出 500
        if depth_key not in ("3", "2", "1") or (
            normalized_paths is not None and not any(len(p) >= 2 for p in normalized_paths)
        ):
            if cached_depth and "prob" in cached_depth:
                p_orig = cached_depth["prob"]
            else:
                model, _ = state.get_or_load_model(depth_key)
                p_orig = _compute_pair_prob(model, depth_key, uid, item_id)
            logger.info(f"無路徑或基準模型探針回傳: 使用者 {uid}, 食譜 {item_id}, 深度 {depth_key}")
            return _build_no_path_probe_response(uid, item_id, depth_key, mode, p_orig)

        if not normalized_paths and cached_depth and cached_depth.get("recommended_item_id") == item_id:
            cached_probe = _lookup_cached_probe_response(uid, item_id, depth_key, mode, cached_depth)
            if cached_probe is not None:
                return cached_probe

        model, explainer = state.get_or_load_model(depth_key)
        if model is None or state.indices is None:
            raise HTTPException(status_code=500, detail="模型或圖結構未載入")

        p_orig = _compute_pair_prob(model, depth_key, uid, item_id)
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

        important_edges = _collect_important_edges(
            uid, item_id, depth_key, normalized_paths, cached_depth, explainer
        )
        if not important_edges:
            logger.info(f"無可遮蔽邊緣: 使用者 {uid}, 食譜 {item_id}")
            return _build_no_path_probe_response(uid, item_id, depth_key, mode, p_orig)

        hashed_important = [n1 * state.num_nodes + n2 for n1, n2 in important_edges]
        hashed_tensor = torch.tensor(hashed_important, device=state.device)
        mask_important = torch.isin(state.edge_hash, hashed_tensor)
        mask_self = state.src == state.dst

        return _evaluate_masked_subgraph(
            model, uid, item_id, depth_key, mode, p_orig, mask_important, mask_self, cache_key
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"探針端點執行異常: 使用者 {req.user_id}, 食譜 {req.item_id}, 模式 {req.mode}: {exc}")
        _clear_accelerator_cache()
        raise HTTPException(status_code=500, detail=f"探針運算異常: {str(exc)}")


# 掛載前端網頁靜態資源
web_dir = os.path.join(current_dir, "web")
if not os.path.exists(web_dir):
    web_dir = os.path.join(project_root, "demo", "web")
if os.path.exists(web_dir):
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="static")


def run():
    parser = argparse.ArgumentParser(description="KGAT 展示推論伺服器")
    parser.add_argument(
        "--port",
        type=int,
        default=get_default_port(),
        help="伺服器監聽端口，預設讀取環境變數 PORT 或使用 8501",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="伺服器監聽位址")
    args = parser.parse_args()

    if args.port == FORBIDDEN_PORT or args.port <= 0 or args.port > 65535:
        print(f"錯誤：為防止連接埠衝突或範圍異常，嚴禁使用 {args.port} 端口，自動重設為 {DEFAULT_PORT}")
        args.port = DEFAULT_PORT
    state.port = args.port

    print(f"啟動展示推論伺服器：http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    run()
