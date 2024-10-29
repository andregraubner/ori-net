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

train, test = get_split("plasmid_taxonomy_islands")
#train, test = get_split("plasmid_random")
_, plasmid_test = get_split("plasmid_taxonomy_islands")
#test_loader = DataLoader(test, batch_size=1, shuffle=False)
#plasmid_test_loader = DataLoader(plasmid_test, batch_size=1, shuffle=False)

# Initialize some hyperparameters
total_steps = 0
grad_acc = 32
n_epochs = 3
lr = 1e-4
warmup=10

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
def evaluate(test_set, path="out.jpg"): 
    flow.eval()
    losses = []
    ious = []

    nrows = len(test_set)
    ncols = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 4 * nrows))

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    
    for (token_ids, labels, row), ax in tqdm(zip(test_loader, axes)): 

        start, end = labels
        
        labels = torch.stack(labels, axis=1).long().to(device)    
        token_ids = token_ids.cuda()

        # had (1, n, 1) here before
        targets = torch.zeros((1, token_ids.shape[1]), dtype=torch.float32, device=device)
        targets[0, labels[0,0]:labels[0,1]+1] = 1  

        targets = targets.unsqueeze(1)
        #targets = F.interpolate(targets, scale_factor=(0.1), mode='linear')
        targets = rearrange(targets, "b c s -> b s c")

        with autocast():
            with torch.inference_mode():
                loss = flow(data=targets, cond=token_ids)
                preds = flow.sample(cond=token_ids, data_shape=(targets.shape[1], 1), batch_size=8, steps=2)
                # calculate IoU
                single_pred = preds[0,:,0]
                ensemble = (preds > 0.5).float().mean(dim=0)[:,0]

        length = ensemble.shape[0]
        hard_preds = (ensemble > 0.5).float()
        
        overlap = hard_preds * (targets[0,:,0] > 0.5).float()
        union = hard_preds + (targets[0,:,0] > 0.5).float()
        iou = overlap.sum() / float(union.sum())

        #ax.plot(range(1,length+1), preds[:,0].to(torch.float32).cpu().numpy(), color="cornflowerblue", linewidth=2, label="predicted ORI start")
        ax.plot(range(1,length+1), single_pred.to(torch.float32).cpu().numpy(), color="lightcoral", linewidth=1, label="single sample ORI prediction")
        ax.plot(range(1,length+1), ensemble.to(torch.float32).cpu().numpy(), color="indianred", linewidth=4, label="ensembled ORI prediction")
        ax.plot(range(1,length+1), targets[0].to(torch.float32).cpu().numpy(), color="royalblue", linewidth=3, label="experimental ORI ")
        #ax.axvline(x=start[0], color='royalblue', linestyle='dashed', linewidth=1, label="experimental ORI start")
        #ax.axvline(x=end[0], color='indianred', linestyle='dashed', linewidth=1, label="experimental ORI end")

        ax.grid(True)
        
        ax.title.set_text(str(iou.item()) + ", " + row["Organism"][0])
        ax.legend()        
        losses.append(loss.item())
        ious.append(iou.item())

    plt.savefig(path)
    plt.close()
    wandb.log({
        f"test loss ({path})": np.mean(losses),
        f"test IoU ({path})": np.mean(ious)
    }, step=total_steps)
    print(np.mean(losses), np.mean(ious))

# Training loops
    
for i in range(1):

    # Create new model
    model = OriFlow().to(device)
    
    flow = RectifiedFlow(
        model=model,
        #loss_fn="pseudo_huber",
    )

    weights = torch.load("weights/pretrained_35000.pth")
    flow.load_state_dict(weights)

    # Create random subset for bootstrapping
    indices = random.sample(range(len(train)), int(1.0 * len(train)))
    train_subset = torch.utils.data.Subset(train, indices)
    train_loader = DataLoader(train_subset, batch_size=1, shuffle=True)

    optimizer = torch.optim.AdamW(flow.parameters(), lr=lr, weight_decay=0.0)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_training_steps=len(train_loader)*n_epochs//grad_acc, 
        num_warmup_steps=warmup
    )
    scaler = GradScaler()

    losses = []

    for epoch in range(n_epochs):

        for step, (token_ids, labels, row) in enumerate(tqdm(train_loader)):
            flow.train()

            labels = torch.stack(labels, axis=1).long().to(device)
            token_ids = token_ids.to(device)

            targets = torch.zeros((1, token_ids.shape[1]), dtype=torch.float32, device=device)
            targets[0, labels[0,0]:labels[0,1]+1] = 1 

            # resizing here
            targets = targets.unsqueeze(1)
            #targets = F.interpolate(targets, scale_factor=(0.1), mode='linear')
            targets = rearrange(targets, "b c s -> b s c")
                        
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

            if total_steps % 1000 == 0:
                evaluate(plasmid_test, "plasmid_test.jpg")
                torch.save(flow.state_dict(), f"weights/finetuned_{total_steps}.pth")
                print("saved model!")

    evaluate(plasmid_test, "plasmid_test.jpg")
    torch.save(flow.state_dict(), f"weights/finetuned_{total_steps}.pth")
    print("saved model!")
    #evaluate()