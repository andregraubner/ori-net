import os
import torch
import numpy as np
from tqdm import tqdm
import random

from model import OriNet

from data import get_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import wandb

wandb.init(
    project="ori-net",
)
device = "cuda:1"

# Initialize dataset
# We use batch size 1 to not have to deal with bidirectional Mamba padding
#dataset = OriDataset("./DoriC12.1/")
#train, test = torch.utils.data.random_split(dataset, [0.9, 0.1], torch.Generator().manual_seed(42))
train, test = get_split("plasmid_random")
test_loader = DataLoader(test, batch_size=1, shuffle=False)

# Initialize model
base_model = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

# Initialize some hyperparameters
total_steps = 0
grad_acc = 1
n_epochs = 1

# Evaluation loop to run after every epoch
# Calculates test loss and saves a graph visualizing predictions to 'figure.jpg'
def evaluate(): 
    model.eval()
    losses = []

    nrows = len(test)
    ncols = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 4 * nrows))

    for (seqs, labels, row), ax in zip(test_loader, axes): 

        start, end = labels
        
        labels = torch.stack(labels, axis=1).long().to(device)

        token_ids = torch.tensor(
            tokenizer.encode(seqs[0]),
            dtype=torch.int,
        ).to(device).unsqueeze(0).long()
        

        with torch.no_grad():
            preds = model(token_ids)[:,:-1]
            loss = F.cross_entropy(preds[:,:,0], labels[:,0]) + F.cross_entropy(preds[:,:,1], labels[:,1])
 
        preds = F.softmax(preds, dim=1)

        preds = preds[0]
        length = preds.shape[0]

        ax.plot(range(1,length+1), preds[:,0].to(torch.float32).cpu().numpy(), color="cornflowerblue", linewidth=2, label="predicted ORI start")
        ax.plot(range(1,length+1), preds[:,1].to(torch.float32).cpu().numpy(), color="lightcoral", linewidth=2, label="predicted ORI end")
        ax.axvline(x=start[0], color='royalblue', linestyle='dashed', linewidth=1, label="experimental ORI start")
        ax.axvline(x=end[0], color='indianred', linestyle='dashed', linewidth=1, label="experimental ORI end")

        ax.grid(True)

        ax.title.set_text(row["Organism"][0] + " / " + row["Class"][0])
        ax.legend()        
        losses.append(loss.item())

    plt.savefig("figure_split.jpg")
    plt.close()
    wandb.log({"test loss": np.mean(losses)}, step=total_steps)

# Training loops
for i in range(10):

    # Create new model
    model = OriNet(base_model).to(device)

    # Create random subset for bootstrapping
    indices = random.sample(range(len(train)), int(0.5 * len(train)))
    train_subset = torch.utils.data.Subset(train, indices)
    train_loader = DataLoader(train_subset, batch_size=1, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_training_steps=len(train_loader)*n_epochs//grad_acc, 
        num_warmup_steps=50
    )
    scaler = GradScaler()

    losses = []
    for epoch in range(n_epochs):

        for step, (seqs, labels, row) in enumerate(tqdm(train_loader)):
            model.train()

            labels = torch.stack(labels, axis=1).long().to(device)

            token_ids = torch.tensor(
                tokenizer.encode(seqs[0]),
                dtype=torch.int,
            ).to(device).unsqueeze(0).long()

            with autocast():
                preds = model(token_ids)[:,:-1]
                loss = F.cross_entropy(preds[:,:,0], labels[:,0]) + F.cross_entropy(preds[:,:,1], labels[:,1])
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
                losses = []

            total_steps += 1

    torch.save(model.state_dict(), f"weights/model_{i}.pth")
    print("saved model!")
        
    #evaluate()