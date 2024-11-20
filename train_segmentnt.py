import os
import torch
import numpy as np
from tqdm import tqdm
import random
from transformers import AutoTokenizer, AutoModel
import torch
from einops import rearrange
from torch import nn

from data import get_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from focal_loss import FocalLoss
import sklearn
import wandb

import torch
from torch.autograd import Variable
from torch import nn
import numpy as np
import torch.nn.functional as F
from pytorch_optimizer import Shampoo

def jaccard_loss(preds, true, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = preds.shape[1]
    true_1_hot = F.one_hot(true)
    true_1_hot = true_1_hot.type(preds.type())
    intersection = torch.sum(preds * true_1_hot, dim=0)
    cardinality = torch.sum(preds + true_1_hot, dim=0)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)

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

wandb.init(
    project="ori-net",
)
device = "cuda:0"

# Initialize dataset
# We use batch size 1 to not have to deal with bidirectional Mamba padding
#dataset = OriDataset("./DoriC12.1/")
#train, test = torch.utils.data.random_split(dataset, [0.9, 0.1], torch.Generator().manual_seed(42))
train, test = get_split("plasmid_taxonomy_islands", max_length=12000)
test_loader = DataLoader(test, batch_size=1, shuffle=False)

# Initialize model
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/segment_nt_multi_species", trust_remote_code=True)

# Initialize some hyperparameters
total_steps = 0
grad_acc = 1
n_epochs = 20
warmup = 1000

criterion = FocalLoss(gamma=2)

# Evaluation loop to run after every epoch
# Calculates test loss and saves a graph visualizing predictions to 'figure.jpg'
def evaluate(): 
    model.eval()
    losses = []
    ious = []

    nrows = len(test)
    ncols = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 4 * nrows))

    for (seqs, labels, row), ax in zip(test_loader, axes): 

        start, end = labels
        
        labels = torch.stack(labels, axis=1).long().to(device)

        max_length = 2001
        tokens = tokenizer.batch_encode_plus(seqs, return_tensors="pt", padding="max_length", max_length=max_length)["input_ids"].to(device)
        if tokens.shape[1] != 2001:
            continue

        attention_mask = tokens != tokenizer.pad_token_id

        with torch.no_grad():
            with autocast():
                preds = model(tokens, attention_mask)[:,:,0]
                preds = F.softmax(preds, dim=2)
                
                targets = torch.zeros((1, preds.shape[1]), dtype=torch.long, device=device)
                targets[0, labels[0,0]:labels[0,1] + 1] = 1 
                
                loss = jaccard_loss(
                    rearrange(preds, "b s c -> (b s) c"), 
                    rearrange(targets, "b s -> (b s)"),
                )

        preds = preds[0,:,1]
        length = preds.shape[0]

        y_pred = (preds >= 0.5).float()
        
        iou = sklearn.metrics.jaccard_score(targets.cpu().numpy().flatten(), y_pred.cpu().numpy().flatten())

        ious.append(iou)

        ax.plot(range(1,length+1), preds.to(torch.float32).cpu().numpy(), color="lightcoral", linewidth=2, label="predicted ORI")
        ax.plot(range(1,length+1), targets[0].to(torch.float32).cpu().numpy(), color="royalblue", linewidth=2, label="experimental ORI ")

        ax.grid(True)

        ax.title.set_text(str(iou.item()) + ", " + row["Organism"][0])
        ax.legend()        
        losses.append(loss.item())

    print(np.mean(losses))

    plt.savefig("figure_split.jpg")
    plt.close()
    wandb.log({
        f"test loss": np.mean(losses),
        f"test IoU": np.mean(ious)
    }, step=total_steps)

# Training loops
for i in range(1):

    # Create new model
    model = OriNetNTSeg().to(device)

    # Create random subset for bootstrapping
    indices = random.sample(range(len(train)), int(1.0 * len(train)))
    train_subset = torch.utils.data.Subset(train, indices)
    train_loader = DataLoader(train_subset, batch_size=1, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6) 
    #optimizer = Shampoo(model.parameters(), lr=1e-5)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_training_steps=len(train_loader)*n_epochs//grad_acc, 
        num_warmup_steps=warmup
    )
    scaler = GradScaler()

    losses = []
    evaluate()
    for epoch in range(n_epochs):

        for step, (seqs, labels, row) in enumerate(tqdm(train_loader)):
            model.train()

            labels = torch.stack(labels, axis=1).long().to(device)

            max_length = 2001
            tokens = tokenizer.batch_encode_plus(seqs, return_tensors="pt", padding="max_length", max_length=max_length)["input_ids"].to(device)

            if tokens.shape[1] != 2001:
                continue

            attention_mask = tokens != tokenizer.pad_token_id

            with autocast():
                preds = model(tokens, attention_mask)[:,:,0]
                preds = F.softmax(preds, dim=2)

                targets = torch.zeros((1, preds.shape[1]), dtype=torch.long, device=device)
                targets[0, labels[0,0]:labels[0,1] + 1] = 1 

                loss = jaccard_loss(
                    rearrange(preds, "b s c -> (b s) c"), 
                    rearrange(targets, "b s -> (b s)"),
                )
                loss /= grad_acc

            scaler.scale(loss).backward()
            losses.append(loss.item() * grad_acc)
            
            if total_steps % grad_acc == 0 and total_steps > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                wandb.log({
                    "train loss": np.mean(losses),
                    "lr": optimizer.param_groups[0]['lr']
                }, step=total_steps)
                print(np.mean(losses))
                losses = []
                
            total_steps += 1

        evaluate()
        
    torch.save(model.state_dict(), f"weights/model_{i}.pth")
    print("saved model!")
        
    #evaluate()