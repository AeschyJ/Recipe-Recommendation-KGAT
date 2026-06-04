import torch
import torch.nn as nn

class LightGCN(nn.Module):
    """
    LightGCN 模型。
    放棄權重矩陣跟非線性激活函數，純粹在二分圖 (Bipartite Graph) 
    上進行訊息平滑傳遞的 GNN 協同過濾典範。
    """
    def __init__(self, n_users, n_items, embed_dim=64, layers=3):
        super(LightGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.layers = layers
        
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.item_embed = nn.Embedding(n_items, embed_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.item_embed.weight)
        
    def _get_norm_adj(self, indices, num_nodes):
        """ 計算 D^{-0.5} * A * D^{-0.5} 的邊權重 """
        src, dst = indices[0], indices[1]
        
        deg = torch.zeros(num_nodes, dtype=torch.float32, device=indices.device)
        deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))
        
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        edge_weight = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
        return edge_weight
        
    def forward(self, indices, num_nodes, user_ids, pos_item_ids, neg_item_ids=None):
        # 推論出聚合後的 Embedding 特徵
        final_embed = self.get_final_embeddings(indices, num_nodes)
        
        u_embed = final_embed[user_ids]
        pos_i_embed = final_embed[self.n_users + pos_item_ids]
        
        pos_scores = torch.sum(u_embed * pos_i_embed, dim=1)
        
        if neg_item_ids is not None:
            neg_i_embed = final_embed[self.n_users + neg_item_ids]
            neg_scores = torch.sum(u_embed * neg_i_embed, dim=1)
            return pos_scores, neg_scores
            
        return pos_scores

    def get_final_embeddings(self, indices, num_nodes):
        """
        全圖 Message Passing
        """
        all_embed = torch.cat([self.user_embed.weight, self.item_embed.weight], dim=0)
        embs = [all_embed]
        
        edge_weight = self._get_norm_adj(indices, num_nodes)
        src, dst = indices[0], indices[1]
        
        for layer in range(self.layers):
            current_dtype = all_embed.dtype
            device = all_embed.device
            
            # ★ XPU bfloat16 index_add_ 效能 Bug Workaround
            # 在 Intel XPU 上，bfloat16 的 index_add_ 會慢 140 倍，因此將聚合區段強制轉為 float32
            with torch.autocast(device_type=device.type, enabled=False):
                msg = all_embed[src].float() * edge_weight.unsqueeze(1).float()
                
                new_embed = torch.zeros(
                    all_embed.shape[0], all_embed.shape[1], device=device, dtype=torch.float32
                )
                new_embed.index_add_(0, dst, msg)
                
            all_embed = new_embed.to(current_dtype)
            embs.append(all_embed)
            
        return torch.mean(torch.stack(embs, dim=0), dim=0)
