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

device = "cuda:0"

train, test = get_split("plasmid_taxonomy_islands", max_length=10000000)
test_loader = DataLoader(test, batch_size=1, shuffle=False)

model = OriNetNTSeg().to(device)
weights = torch.load("weights/model_0.pth")
model.load_state_dict(weights)

window_size = 6000*5
stride = 1
max_length = 5000+1

# Initialize model
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/segment_nt_multi_species", trust_remote_code=True)

# Evaluation loop to run after every epoch
# Calculates test loss and saves a graph visualizing predictions to 'figure.jpg'
def evaluate(): 
    model.eval()
    ious = []

    nrows = len(test)
    ncols = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 4 * nrows))

    for (seqs, labels, row), ax in zip(test_loader, axes): 

        start, end = labels
        sequence = seqs[0]
        
        labels = torch.stack(labels, axis=1).long().to(device)

        n = len(sequence)
        counts = torch.zeros(n,1).to(device)
        outputs = torch.zeros(n,2).to(device)
        if len(sequence) > window_size:
            for i in tqdm(range(0, n - window_size + 1, stride)):
                chunk = sequence[i:i + window_size]
    
                tokens = tokenizer.batch_encode_plus([chunk], return_tensors="pt", padding="max_length", max_length=max_length)["input_ids"].to(device)
                if tokens.shape[1] != max_length:
                    print(tokens.shape)
                    print("oof")
                    continue
    
                attention_mask = tokens != tokenizer.pad_token_id
    
                with torch.inference_mode():
                    with autocast():
                        preds = model(tokens, attention_mask)[:,:,0]
                        preds = preds[0,:]
                        preds = F.softmax(preds, dim=1)
                        
                outputs[i:i + window_size] += preds
                counts[i:i + window_size] += 1
            
            outputs /= counts
            preds = outputs[...,1] 
        else:
            tokens = tokenizer.batch_encode_plus([sequence], return_tensors="pt", padding="max_length", max_length=max_length)["input_ids"].to(device)
            if tokens.shape[1] != max_length:
                print(tokens.shape)
                print("oof")
                continue

            attention_mask = tokens != tokenizer.pad_token_id

            with torch.inference_mode():
                with autocast():
                    preds = model(tokens, attention_mask)[:,:,0]
                    outputs = preds[0,:]
                    preds = F.softmax(outputs, dim=1)[...,1] 
           
        targets = torch.zeros(len(preds), dtype=torch.long, device=device)
        targets[labels[0,0]:labels[0,1] + 1] = 1 
        
        y_pred = (preds >= 0.5).float()
        
        iou = sklearn.metrics.jaccard_score(targets.cpu().numpy().flatten(), y_pred.cpu().numpy().flatten())

        ious.append(iou)
        print(np.mean(ious), iou)

        length = preds.shape[0]
        ax.plot(range(1,length+1), preds.to(torch.float32).cpu().numpy(), color="lightcoral", linewidth=2, label="predicted ORI")
        ax.plot(range(1,length+1), targets.to(torch.float32).cpu().numpy(), color="royalblue", linewidth=2, label="experimental ORI ")

        ax.grid(True)

        ax.title.set_text(str(iou.item()) + ", " + row["Organism"][0])
        ax.legend()        

    print(np.mean(ious))

    plt.savefig("figure_split.jpg")
    plt.close()

evaluate()