import os
import torch
import numpy as np
from tqdm import tqdm
import random
from transformers import AutoTokenizer, AutoModel
import torch
from einops import rearrange


from model import OriNetNTSeg

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

wandb.init(
    project="ori-net",
)
device = "cuda:0"

# Initialize dataset
# We use batch size 1 to not have to deal with bidirectional Mamba padding
#dataset = OriDataset("./DoriC12.1/")
#train, test = torch.utils.data.random_split(dataset, [0.9, 0.1], torch.Generator().manual_seed(42))
train, test = get_split("plasmid_taxonomy_islands")
test_loader = DataLoader(test, batch_size=1, shuffle=False)

# Initialize model
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/segment_nt_multi_species", trust_remote_code=True)

# Initialize some hyperparameters
total_steps = 0
grad_acc = 1
n_epochs = 15
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
                
                loss = criterion(
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

    #optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

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

                loss = criterion(
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