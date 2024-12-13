import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

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
        self.model.num_features = 1
        self.model.fc = nn.Linear(in_features=1024, out_features=6 * 2)

    def forward(self, tokens, attention_mask):

        preds = self.model(
                    tokens,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                ).logits
        return preds