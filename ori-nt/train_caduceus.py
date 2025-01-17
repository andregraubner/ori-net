import os
import torch
import numpy as np
from tqdm import tqdm
import random
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from einops import rearrange
from data import get_split
from torch.utils.data import DataLoader
import torch.nn.functional as F
import seaborn as sns
import sklearn
import wandb
from utils import jaccard_loss
from model import OriCaduceus
from torch.amp import autocast, GradScaler

# Initialize some hyperparameters
config = {
    "batch_size": 1,
    "lr": 1e-4,
    "grad_acc": 1,
    "n_epochs": 15,
    "warmup_steps": 1000,
    "max_seq_length": 2001, # 1000 + 1 CLS token
    "device": "cuda",
    "seed": 42
}
device = config["device"]
untokenized_seq_len = (config["max_seq_length"] - 1) * 6

wandb.init(
    project="ORI-caduceus",
)

# Set random seeds
os.environ['PYTHONHASHSEED'] = str(config["seed"])
torch.manual_seed(config["seed"])
torch.cuda.manual_seed(config["seed"])
torch.cuda.manual_seed_all(config["seed"])
np.random.seed(config["seed"])
random.seed(config["seed"])

# Initialize model
model = OriCaduceus().to(config["device"])
#model = torch.compile(model)
tokenizer = AutoTokenizer.from_pretrained("kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16", trust_remote_code=True)

# Create train and test sets
train, test = get_split("plasmid_taxonomy_islands", max_length=untokenized_seq_len)
train_loader = DataLoader(train, batch_size=config["batch_size"], shuffle=True, collate_fn=train.collate_caduceus)
test_loader = DataLoader(test, batch_size=1, shuffle=False, collate_fn=test.collate_caduceus)

optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"]) 
scheduler = get_cosine_schedule_with_warmup(
    optimizer, 
    num_training_steps=len(train_loader) * config["n_epochs"] // config["grad_acc"], 
    num_warmup_steps=config["warmup_steps"]
)
scaler = GradScaler("cuda")    

total_steps = 0
losses = []
for epoch in range(config["n_epochs"]):

    model.train()
    for step, (seqs, tokens, targets, metadata) in enumerate(tqdm(train_loader)):

        tokens, targets = tokens.to(device), targets.to(device)

        #if tokens.shape[1] != config["max_seq_length"]:
        #   print("Skipping sample")
        #    continue
        
        with autocast("cuda"):
            preds = model(tokens)[:,:-1] # Remove CLS token
            preds = F.softmax(preds, dim=2)

            loss = jaccard_loss(
                rearrange(preds, "b s c -> (b s) c"), 
                rearrange(targets, "b s -> (b s)"),
            )
            loss /= config["grad_acc"]

        scaler.scale(loss).backward()
        losses.append(loss.item() * config["grad_acc"])
        
        if total_steps % config["grad_acc"] == 0 and total_steps > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            wandb.log({
                "train loss": np.mean(losses),
                "lr": optimizer.param_groups[0]['lr']
            }, step=total_steps)
            losses = []
            
        total_steps += 1

    # Evaluate
    model.eval()
    losses = []
    ious = []

    for seqs, tokens, targets, metadata in test_loader: 

        tokens, targets = tokens.to(device), targets.to(device)

        #if tokens.shape[1] != config["max_seq_length"]:
        #    print("Skipping sample")
        #    continue

        with torch.inference_mode():
            with autocast("cuda"):
                preds = model(tokens)[:,:-1] # Remove CLS token
                preds = F.softmax(preds, dim=2)
                
                loss = jaccard_loss(
                    rearrange(preds, "b s c -> (b s) c"), 
                    rearrange(targets, "b s -> (b s)"),
                )
                losses.append(loss.item())

        y_pred = (preds >= 0.5)[...,1].float()

        for i, seq in enumerate(seqs):
            l = len(seq)
            iou = sklearn.metrics.jaccard_score(targets[i,:l].cpu().numpy().flatten(), y_pred[i,:l].cpu().numpy().flatten())
            ious.append(iou)

    print(np.mean(losses))
    print(np.mean(ious))

    wandb.log({
        f"test loss": np.mean(losses),
        f"test IoU": np.mean(ious)
    }, step=total_steps)

    losses = []
    
    torch.save(model.state_dict(), f"weights/ori_caduceus.pth")
    print("saved model!")