import torch
import torch.nn as nn

class BPRMF(nn.Module):
    """
    傳統的矩陣分解協同過濾模型 (BPR-MF)。
    作為推薦系統基準 (Baseline)，不包含任何 GNN 或 Knowledge Graph 資訊。
    """
    def __init__(self, n_users, n_items, embed_dim=64):
        super(BPRMF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.item_embed = nn.Embedding(n_items, embed_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.item_embed.weight)
        
    def forward(self, indices, num_nodes, user_ids, pos_item_ids, neg_item_ids=None):
        """
        支援統一個訓練迴圈呼叫介面，但在此模型中 indices 與 num_nodes 會被忽略。
        """
        u_embed = self.user_embed(user_ids)
        pos_i_embed = self.item_embed(pos_item_ids)
        
        pos_scores = torch.sum(u_embed * pos_i_embed, dim=1)
        
        if neg_item_ids is not None:
            neg_i_embed = self.item_embed(neg_item_ids)
            neg_scores = torch.sum(u_embed * neg_i_embed, dim=1)
            return pos_scores, neg_scores
            
        return pos_scores
        
    def get_final_embeddings(self, indices=None, num_nodes=None):
        """
        為能使用 Dot Product 高速測試評估，提供快速萃取 Embedding 的方法。
        """
        return torch.cat([self.user_embed.weight, self.item_embed.weight], dim=0)
