from torch import nn
from transformers import AutoModel

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class OriNet(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.caduceus = AutoModel.from_pretrained(base_model, trust_remote_code=True)
        self.ff = FeedForward(dim=512, hidden_dim=1024*8, dropout=0.2)
        self.head = nn.Linear(512, 2, bias=False)

    def forward(self, inputs):

        x = self.caduceus(inputs)["last_hidden_state"]
        x = self.ff(x)
        x = self.head(x)
        
        return x