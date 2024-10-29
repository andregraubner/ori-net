import os
import torch
import numpy as np
from tqdm import tqdm
import random
from transformers import AutoTokenizer, AutoModel
import torch
from einops import rearrange
from Bio import SeqIO

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

model = OriNetNTSeg().to(device)
weights = torch.load("weights/model_0.pth")
model.load_state_dict(weights)

window_size = 6000*5
stride = 5000
max_length = 5000+1

fname = "p1"

p1 = list(SeqIO.parse(f"helen/{fname}.fa", "fasta"))[0]

# Initialize model
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/segment_nt_multi_species", trust_remote_code=True)

# Evaluation loop to run after every epoch
# Calculates test loss and saves a graph visualizing predictions to 'figure.jpg'
model.eval()
ious = []

nrows = 1
ncols = 1
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 4 * nrows))
ax = axes

sequence = str(p1.seq)

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
            quit()

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
        quit()

    attention_mask = tokens != tokenizer.pad_token_id

    with torch.inference_mode():
        with autocast():
            preds = model(tokens, attention_mask)[:,:,0]
            outputs = preds[0,:]
            preds = F.softmax(outputs, dim=1)[...,1] 

torch.save(preds, f"preds_{fname}.pth")

y_pred = (preds >= 0.5).float()


length = preds.shape[0]
ax.plot(range(1,length+1), preds.to(torch.float32).cpu().numpy(), color="lightcoral", linewidth=2, label="predicted ORI")

ax.grid(True)

ax.title.set_text(f"{fname}")
ax.legend()        

plt.savefig(f"helen_{fname}.jpg")
plt.close()