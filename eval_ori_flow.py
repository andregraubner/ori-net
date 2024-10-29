import os
import torch
import numpy as np
from tqdm import tqdm
import random
from torch import nn

from sklearn.metrics import precision_recall_curve, auc
import numpy as np
import matplotlib.pyplot as plt


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

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

train, test = get_split("plasmid_taxonomy_islands")
test_loader = DataLoader(test, batch_size=1, shuffle=False)

model = OriFlow().to(device)

flow = RectifiedFlow(
        model=model,
        #loss_fn="pseudo_huber",
    )

weights = torch.load("weights/finetuned_3015.pth")
flow.load_state_dict(weights)

def evaluate(): 
    model.eval()
    losses = []
    ious = []
    y_trues = []
    y_preds = []

    nrows = len(test)
    ncols = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 4 * nrows))
    iterations = 0

    for (token_ids, labels, row), ax in tqdm(zip(test_loader, axes)): 

        start, end = labels
        
        labels = torch.stack(labels, axis=1).long().to(device)    
        token_ids = token_ids.cuda()

        targets = torch.zeros((1, token_ids.shape[1], 1), dtype=torch.float32, device=device)
        targets[0, labels[0,0]:labels[0,1]+1] = 1  

        with autocast():
            with torch.inference_mode():
                loss = flow(data=targets, cond=token_ids)
                preds = flow.sample(cond=token_ids, data_shape=(token_ids.shape[1], 1), batch_size=16, steps=4)
                # calculate IoU
                single_pred = preds[0,:,0]
                #ensemble = (preds > 0.5).float().mean(dim=0)[:,0]
                ensemble = preds.mean(dim=0)[:,0]

        length = ensemble.shape[0]
        hard_preds = (ensemble >= 0.6).float()
        
        overlap = hard_preds * targets[0,:,0]
        union = hard_preds + targets[0,:,0]
        iou = overlap.sum() / float(union.sum())

        y_true = targets[0,:,0].flatten().cpu().numpy()  # Assuming targets is a torch tensor
        y_pred = ensemble.flatten().cpu().numpy()

        y_trues.append(y_true)
        y_preds.append(y_pred)

        #ax.plot(range(1,length+1), preds[:,0].to(torch.float32).cpu().numpy(), color="cornflowerblue", linewidth=2, label="predicted ORI start")
        #ax.plot(range(1,length+1), single_pred.to(torch.float32).cpu().numpy(), color="lightcoral", linewidth=1, label="single sample ORI prediction")
        ax.plot(range(1,length+1), hard_preds.to(torch.float32).cpu().numpy(), color="indianred", linewidth=4, label="ensembled ORI prediction")
        ax.plot(range(1,length+1), targets[0].to(torch.float32).cpu().numpy(), color="royalblue", linewidth=3, label="experimental ORI ")
        #ax.axvline(x=start[0], color='royalblue', linestyle='dashed', linewidth=1, label="experimental ORI start")
        #ax.axvline(x=end[0], color='indianred', linestyle='dashed', linewidth=1, label="experimental ORI end")

        ax.grid(True)
        
        ax.title.set_text(str(iou.item()) + ", " + row["Organism"][0])
        ax.legend()        
        losses.append(loss.item())
        ious.append(iou.item())
        iterations += 1

        #if iterations == 10:
        #    break

    print("HI")

    plt.savefig("evaluation.jpg")
    plt.close()    

    # Calculate precision-recall curve
    print("Concatenating")
    y_true = np.concatenate(y_trues)
    print(y_true.shape)
    y_pred = np.concatenate(y_preds)
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    
    # Calculate AUC-PR
    auc_pr = auc(recall, precision)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    
    plt.text(0.5, 0.5, f'AUC-PR: {auc_pr:.3f}', 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.savefig("Precision-Recall Curve")
    
    print(np.mean(losses), np.mean(ious))

evaluate()