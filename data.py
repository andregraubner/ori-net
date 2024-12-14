from torch.utils.data import Dataset
from Bio import SeqIO
import pandas as pd
import random
import torch
from sklearn.model_selection import train_test_split
import random
from transformers import AutoTokenizer
from utils import shift_ori

nt_tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/segment_nt_multi_species", trust_remote_code=True)

class OriDataset(Dataset):

    def __init__(self, annotations, sequences, max_length):

        self.annotations = annotations
        self.sequences = sequences
        self.max_length = max_length

        self.sequences = [seq for seq in self.sequences if seq.id in self.annotations.index]        
        self.sequences = [seq for seq in self.sequences if all(base in ["A", "C", "G", "T"] for base in seq)]
        
    def __getitem__(self, idx):

        data = self.sequences[idx]
        seq = str(data.seq)
        metadata = self.annotations.loc[data.id]

        # Convert indices from 1-indexed (DoriC) to 0-indexed (tensors)
        start = metadata["OriC start"] - 1
        end = metadata["OriC end"] - 1

        # In reality, there are just a few instances where start == len(seq). 
        # In this case, this code sets it to 0
        start = start % len(seq)
        end = end % len(seq)    

        old_slen = len(seq)

        seq, start, end = shift_ori(seq, start, end, max_length=self.max_length)
        
        return seq, (start, end), metadata.to_dict()

    def __len__(self):
        return len(self.sequences)

    def collate_nt(self, batch):

        seqs, labels, metadata = zip(*batch)
        bs = len(labels)
        
        # Create one-hot target tensor
        targets = torch.zeros(bs, self.max_length, dtype=torch.long)
        for i, (start, end) in enumerate(labels):
            targets[i, start:end + 1] = 1 
        
        tokens = nt_tokenizer.batch_encode_plus(
            seqs,
            return_tensors="pt", 
            padding="max_length", 
            max_length=self.max_length // 6 + 1
        )["input_ids"]
    
        return seqs, tokens, targets, metadata

def get_split(split_name, max_length=None):

    plasmid_annotations = pd.read_csv("DoriC12.1/DoriC12.1_plasmid.csv")
    plasmid_annotations["NC"] = plasmid_annotations["NC"].astype(str) + ".1"

    mask = plasmid_annotations.duplicated('Refseq', keep=False)
    plasmid_annotations = plasmid_annotations[~mask]
    
    plasmid_annotations.set_index('NC', inplace=True)

    # Define the taxonomic levels
    levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species', 'Strain']
    
    # Split the taxonomy column
    plasmid_annotations[levels] = plasmid_annotations['Lineage'].str.split(',', expand=True)
    
    # Trim whitespace from the new columns
    for level in levels:
        plasmid_annotations[level] = plasmid_annotations[level].str.strip()
    
    # Fill NaN values with empty string
    plasmid_annotations[levels] = plasmid_annotations[levels].fillna('')

    class_counts = plasmid_annotations['Class'].value_counts()

    # Create a boolean mask for classes appearing over 25 times
    mask = plasmid_annotations['Class'].map(class_counts) > 25
    
    # Split the dataframe
    train = plasmid_annotations[mask]
    test = plasmid_annotations[~mask]

    sequences = list(SeqIO.parse("/root/autodl-fs/sequences.fasta", "fasta"))
    
    return OriDataset(train, sequences, max_length), OriDataset(test, sequences, max_length)