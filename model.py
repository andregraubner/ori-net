from torch import nn
from transformers import AutoModel, AutoTokenizer
from plasmamba import CaduceusForMaskedLM, CaduceusConfig
import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from TorchCRF import CRF

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
        model.load_state_dict(torch.load("/root/autodl-fs/weights/model_9.pth"))
        self.plasmamba = model
        self.ff = FeedForward(dim=512, hidden_dim=1024*8, dropout=0.2)
        self.head = nn.Linear(512, 2, bias=False)

    def forward(self, inputs):

        x = self.plasmamba(inputs, output_hidden_states=True)["hidden_states"][-1]
        #x = self.ff(x)
        x = self.head(x)
        
        return x

class FinalConv1D(nn.Module):
    """
    Final output block of the 1D-UNET.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_layers: int = 2,
    ):
        """
        Args:
            output_channels: number of output channels.
            activation_fn: name of the activation function to use.
                Should be one of "gelu",
                "gelu-no-approx", "relu", "swish", "silu", "sin".
            num_layers: number of convolution layers.
            name: module name.
        """
        super().__init__()

        self._first_layer = [nn.Conv1d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding="same",
            )]

        self._next_layers = [
            nn.Conv1d(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding="same",
            )
            for _ in range(num_layers-1)
        ]
        self.conv_layers = nn.ModuleList(self._first_layer + self._next_layers)

        self._activation_fn = nn.SiLU()



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            if i < len(self.conv_layers) - 1:
                x = self._activation_fn(x)
        return x

class OriNetNTSeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("InstaDeepAI/segment_nt_multi_species", trust_remote_code=True)
        self.model.unet.final_block = FinalConv1D(
            input_channels=1024,
            output_channels=1024,
            num_layers=2,
        )
        print(self.model.num_features)
        self.model.num_features = 1
        print(self.model.num_features)
        self.model.fc = nn.Linear(in_features=1024, out_features=6 * 2)

    def forward(self, tokens, attention_mask):

        preds = self.model(
                    tokens,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                ).logits
        return preds

class OriNetCRF(nn.Module):
    def __init__(self):
        super().__init__()
        model = CaduceusForMaskedLM(config)
        model.load_state_dict(torch.load("/root/autodl-fs/weights/model_9.pth"))
        self.plasmamba = model
        self.head = nn.Linear(512, 2, bias=False)
        self.crf = CRF(num_labels=2)
        
    def forward(self, inputs, labels=None):

        x = self.plasmamba(inputs, output_hidden_states=True)["hidden_states"][-1]
        logits = self.head(x)
        mask = torch.ones_like(logits[:,:,0]).bool()

        if labels is not None:
            loss = -self.crf(logits, labels, mask)
            return logits, loss

        outputs = self.crf.viterbi_decode(logits, mask)
        return torch.tensor(outputs, device=logits.device).long()

class OriUNet(nn.Module):
    def __init__(self):
        super().__init__()
        model = CaduceusForMaskedLM(config)
        model.load_state_dict(torch.load("/root/autodl-fs/weights/model_9.pth"))
        self.plasmamba = model
        self.unet = Unet1D(
            dim = 32,
            dim_mults = (1, 2, 4, 8),
            channels = 512
        )

    def forward(self, inputs):

        x = self.plasmamba(inputs, output_hidden_states=True)["hidden_states"][-1]
        #x = self.ff(x)
        x = self.head(x)
        
        return x