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

import wandb

device = "cuda:1"

# Initialize dataset
# We use batch size 1 to not have to deal with bidirectional Mamba padding
#dataset = OriDataset("./DoriC12.1/")
#train, test = torch.utils.data.random_split(dataset, [0.9, 0.1], torch.Generator().manual_seed(42))
train, test = get_split("taxonomy_islands")
test_loader = DataLoader(test, batch_size=1, shuffle=False)

# Initialize model
base_model = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

n_models = 10
models = []
for i in range(n_models):
    model = OriNet(base_model).to(device)
    state_dict = torch.load(f"weights/model_{i}.pth")
    model.load_state_dict(state_dict)
    model.eval()
    models.append(model)

# Evaluation loop to run after every epoch
# Calculates test loss and saves a graph visualizing predictions to 'figure.jpg'
def evaluate(): 
    losses = []
    i = 0

    nrows = 10 #len(test)
    ncols = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 4 * nrows))

    for (seqs, labels, row), ax in tqdm(zip(test_loader, axes), total=len(test_loader)): 

        start, end = labels
        
        labels = torch.stack(labels, axis=1).long().to(device)

        token_ids = torch.tensor(
            tokenizer.encode(seqs[0]),
            dtype=torch.int,
        ).to(device).unsqueeze(0).long()
        

        with torch.inference_mode():
            pred_list = []
            for model in models:
                pred_list.append(model(token_ids)[:,:-1])
            preds = torch.cat(pred_list, dim=0).mean(dim=0, keepdims=True)
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

        i += 1
        if i >= 10:
            break

    plt.savefig("figure_split.jpg")
    plt.close()

evaluate()