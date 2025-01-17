import os
import torch
import numpy as np
from tqdm import tqdm
import random
from transformers import AutoTokenizer, AutoModel
import torch
from einops import rearrange
from Bio import SeqIO

from model import OriNT

from data import get_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

import argparse

def main():

    device = "cuda"
    
    # Create parser
    parser = argparse.ArgumentParser(
        description='Run ORI-NT on the provided sequences and save ORI probabilities to numpy file',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Add arguments
    parser.add_argument('-c', '--checkpoint', help='Model checkpoint path')
    parser.add_argument('-i', '--input', help='Input fasta file')
    parser.add_argument('-o', '--output', help='Output numpy file')
    parser.add_argument('-s', '--seed', default=42, help='Seed')
    args = parser.parse_args()

    # Set random seeds
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Load trained model
    model = OriNT().to(device)
    weights = torch.load(args.checkpoint)
    weights = {k.replace("_orig_mod.", ""): v for k, v in weights.items()}
    model.load_state_dict(weights)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/segment_nt_multi_species", trust_remote_code=True)

    n_tokens = 5000 # Sequence length of tokens to show the model at once
    max_length = n_tokens+1 # Special token
    window_size = n_tokens*6 # Tokenizer puts 6 nucleotides into one token
    stride = 5000 # Number or nucleotides in to overlap chunks

    results = {}
    for record in tqdm(list(SeqIO.parse(args.input, "fasta"))):
        sequence_id = record.id
        description = record.description
        sequence = str(record.seq)

        pad_size = min(window_size, len(sequence))

        padded_sequence = sequence[-pad_size:] + sequence + sequence[:pad_size]
    
        n = len(padded_sequence)
        counts = torch.zeros(n,1).to(device)
        outputs = torch.zeros(n,2).to(device)
        
        for i in tqdm(range(0, max(1, n - window_size + 1), stride)):
            ws = min(window_size, n)
            chunk = padded_sequence[i:i + ws]

            tokens = tokenizer.batch_encode_plus(
                [chunk], 
                return_tensors="pt", 
                padding="max_length", 
                max_length=max_length
            )["input_ids"].to(device)
            
            if tokens.shape[1] != max_length:
                print(f"Tokenisation issue... skipping {record}")
                continue

            attention_mask = tokens != tokenizer.pad_token_id
            with torch.inference_mode():
                with autocast():
                    preds = model(tokens, attention_mask)[:,:,0]
                    preds = preds[0,:]
                    preds = F.softmax(preds, dim=1)
            
            outputs[i:i + ws] += preds[:ws]
            counts[i:i + ws] += 1
            
        outputs /= counts
        preds = outputs[...,pad_size:-pad_size,1].cpu().numpy()
        results[record.id] = preds

    np.save(args.output,  results)    

if __name__ == '__main__':
    main()