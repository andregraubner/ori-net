from torch import nn
from transformers import AutoModel, AutoTokenizer
from plasmamba import CaduceusForMaskedLM, CaduceusConfig
import torch

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

config = CaduceusConfig(
    d_model = 512,
    n_layer = 8,
    vocab_size = tokenizer.vocab_size,
    pad_token_id = tokenizer.pad_token_id,

    # Caduceus-specific params
    bidirectional = True,
    bidirectional_strategy = "add",
    bidirectional_weight_tie = True,
    rcps = False
)

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
    def __init__(self):
        super().__init__()
        model = CaduceusForMaskedLM(config)
        model.load_state_dict(torch.load("/root/autodl-fs/model_1.pth"))
        self.plasmamba = model
        self.ff = FeedForward(dim=512, hidden_dim=1024*8, dropout=0.2)
        self.head = nn.Linear(512, 2, bias=False)

    def forward(self, inputs):

        x = self.plasmamba(inputs, output_hidden_states=True)["hidden_states"][-1]
        x = self.ff(x)
        x = self.head(x)
        
        return x