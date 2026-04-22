import torch
import torch.nn as nn

class NFM(nn.Module):
    """
    Neural Factorization Machine (NFM).
    融合了二階特徵交互 (Bi-Interaction Pooling) 以及深層神經網路 (MLP)。
    由於 NFM 以單一數值為預估輸出，而非單純的 Embedding 內積，評估方式較為不同。
    """
    def __init__(self, n_users, n_items, embed_dim=64, hidden_layers=[64, 32]):
        super(NFM, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.item_embed = nn.Embedding(n_items, embed_dim)
        
        # Bi-Interaction 後接的 MLP (Deep Inference)
        mlp_modules = []
        in_dim = embed_dim
        for out_dim in hidden_layers:
            mlp_modules.append(nn.Linear(in_dim, out_dim))
            mlp_modules.append(nn.BatchNorm1d(out_dim))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(0.2))
            in_dim = out_dim
        
        mlp_modules.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*mlp_modules)
        
        # 一次項 (Bias Terms)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.item_embed.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        
    def _compute_score(self, u_emb, i_emb, u_b, i_b):
        # 二次互動項 (Bi-Interaction Pooling)
        bi_interact = u_emb * i_emb
        
        # 深度神經網路學習
        mlp_out = self.mlp(bi_interact).squeeze()
        
        # 線性特徵
        linear_part = u_b.squeeze() + i_b.squeeze() + self.global_bias
        
        return linear_part + mlp_out

    def forward(self, indices, num_nodes, user_ids, pos_item_ids, neg_item_ids=None):
        u_emb = self.user_embed(user_ids)
        u_b = self.user_bias(user_ids)
        
        pos_i_emb = self.item_embed(pos_item_ids)
        pos_i_b = self.item_bias(pos_item_ids)
        pos_scores = self._compute_score(u_emb, pos_i_emb, u_b, pos_i_b)
        
        if neg_item_ids is not None:
            neg_i_emb = self.item_embed(neg_item_ids)
            neg_i_b = self.item_bias(neg_item_ids)
            neg_scores = self._compute_score(u_emb, neg_i_emb, u_b, neg_i_b)
            return pos_scores, neg_scores
            
        return pos_scores

    def evaluate_forward(self, user_ids, item_ids):
        """供驗證用，不依賴預先融合好的 embedding"""
        u_emb = self.user_embed(user_ids)
        u_b = self.user_bias(user_ids)
        
        i_emb = self.item_embed(item_ids)
        i_b = self.item_bias(item_ids)
        return self._compute_score(u_emb, i_emb, u_b, i_b)
