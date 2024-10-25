import os
import torch
import numpy as np
from tqdm import tqdm
import random
from torch import nn

from model import OriNet, OriNetCRF

from data import get_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import wandb

from plasmamba import CaduceusForMaskedLM, CaduceusConfig

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from einops import rearrange

from ori_flow import OriFlow
from rectified_flow import RectifiedFlow


device = "cuda"

wandb.init(
    project="ori-net-tagging",
)

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

train, test = get_split("taxonomy_islands")
test_loader = DataLoader(test, batch_size=1, shuffle=False)

# Initialize some hyperparameters
total_steps = 0
grad_acc = 4
n_epochs = 15

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:  # 'none'
            return focal_loss

# Usage example:
criterion = FocalLoss(alpha=0.1, gamma=5)
weight=torch.tensor([0.1, 1.0], device="cuda", dtype=torch.float32)

# Evaluation loop to run after every epoch
# Calculates test loss and saves a graph visualizing predictions to 'figure.jpg'
def evaluate(): 
    flow.eval()
    losses = []
    ious = []

    nrows = len(test)
    ncols = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 4 * nrows))

    for (token_ids, labels, row), ax in tqdm(zip(test_loader, axes)): 

        start, end = labels
        
        labels = torch.stack(labels, axis=1).long().to(device)    
        token_ids = token_ids.cuda()

        targets = torch.zeros((1, token_ids.shape[1], 1), dtype=torch.float32, device=device)
        targets[0, labels[0,0]:labels[0,1]+1] = 1  

        with torch.inference_mode():
            loss = flow(data=targets, cond=token_ids)
            #preds = model(token_ids)#[:,:-1]
            #loss = F.cross_entropy(
            #    rearrange(preds, "b s c -> (b s) c"), 
            #    rearrange(targets, "b s -> (b s)"),
            #    weight=weight
            #)
            #loss = criterion(
            #    rearrange(preds, "b s c -> (b s) c"), 
            #    rearrange(targets, "b s -> (b s)"),
            #)
            #loss = F.cross_entropy(preds.permute(0,2,1), targets)
 
            #preds = F.softmax(preds, dim=-1)
            preds = flow.sample(cond=token_ids, data_shape=(token_ids.shape[1], 1), steps=4)

        preds = preds[0]
        length = preds.shape[0]

        # calculate IoU
        hard_preds = (preds > 0.5).float()
        overlap = hard_preds * targets[0,:,0]
        union = hard_preds + targets[0,:,0]
        iou = overlap.sum() / float(union.sum())

        #ax.plot(range(1,length+1), preds[:,0].to(torch.float32).cpu().numpy(), color="cornflowerblue", linewidth=2, label="predicted ORI start")
        ax.plot(range(1,length+1), preds.to(torch.float32).cpu().numpy(), color="lightcoral", linewidth=2, label="predicted ORI")
        ax.plot(range(1,length+1), targets[0].to(torch.float32).cpu().numpy(), color="royalblue", linewidth=2, label="experimental ORI ")
        #ax.axvline(x=start[0], color='royalblue', linestyle='dashed', linewidth=1, label="experimental ORI start")
        #ax.axvline(x=end[0], color='indianred', linestyle='dashed', linewidth=1, label="experimental ORI end")

        ax.grid(True)

        #ax.title.set_text(row["Organism"][0] + " / " + row["Class"][0])
        ax.legend()        
        losses.append(loss.item())
        ious.append(iou.item())

    plt.savefig("figure_split.jpg")
    plt.close()
    wandb.log({
        "test loss": np.mean(losses),
        "test IoU": np.mean(ious)
    }, step=total_steps)
    print(np.mean(losses), np.mean(ious))

# Training loops
    
for i in range(1):

    # Create new model
    model = OriFlow().to(device)
    flow = RectifiedFlow(model=model)

    # Create random subset for bootstrapping
    indices = random.sample(range(len(train)), int(1.0 * len(train)))
    train_subset = torch.utils.data.Subset(train, indices)
    train_loader = DataLoader(train_subset, batch_size=1, shuffle=True)

    optimizer = torch.optim.AdamW(flow.parameters(), lr=3e-4, weight_decay=0.0)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_training_steps=len(train_loader)*n_epochs//grad_acc, 
        num_warmup_steps=8
    )
    scaler = GradScaler()

    losses = []

    for epoch in range(n_epochs):

        for step, (token_ids, labels, row) in enumerate(tqdm(train_loader)):
            flow.train()

            labels = torch.stack(labels, axis=1).long().to(device)
            token_ids = token_ids.to(device)

            targets = torch.zeros((1, token_ids.shape[1], 1), dtype=torch.float32, device=device)
            targets[0, labels[0,0]:labels[0,1]+1] = 1 
            
            with autocast():
                
                loss = flow(data=targets, cond=token_ids)
                #preds = model(token_ids) #[:,:-1]
                #loss = F.cross_entropy(
                #    rearrange(preds, "b s c -> (b s) c"), 
                #    rearrange(targets, "b s -> (b s)"),
                #    weight=weight
                #)
                #loss = criterion(
                #    rearrange(preds, "b s c -> (b s) c"), 
                #    rearrange(targets, "b s -> (b s)"),
                #)
                #loss = criterion(preds[:,:,0], labels[:,0]) + criterion(preds[:,:,1], labels[:,1])
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

        if epoch % 1 == 0:
            evaluate()
            torch.save(model.state_dict(), f"weights/flow_newnew_{epoch}.pth")
            print("saved model!")

    evaluate()
    torch.save(model.state_dict(), f"weights/flow_newnew_{epoch}.pth")
    print("saved model!")
    #evaluate()