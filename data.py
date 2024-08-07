from torch.utils.data import Dataset
from Bio import SeqIO
import pandas as pd
import random
import torch

def random_subsequence(sequence, start, end, n):
    """Generate a random subsequence that contains the start and end location"""
    length = len(sequence)
    
    # Step 1: Calculate the effective length of the slice
    if end < start:
        slice_length = length - start + end
    else:
        slice_length = end - start
    
    # Step 2: Generate a random starting index for the subsequence
    if end < start:
        random_start = random.randint(end, start)
    else:
        random_start = random.randint(max(0, start - n + 1), start)
    
    # Step 3: Extract the subsequence of length n starting from the random index
    subsequence = ""
    for i in range(n):
        index = (random_start + i) % length
        subsequence += sequence[index]
    
    # Step 4: Calculate the start and end locations of the slice within the subsequence
    subsequence_start = (start - random_start) % n
    subsequence_end = (end - random_start) % n
    
    return subsequence, subsequence_start, subsequence_end

class OriDataset(Dataset):

    def __init__(self, path):

        self.annotations = pd.read_csv(path + "/annotations.csv")
        mask = self.annotations.duplicated('Refseq', keep=False)
        self.annotations = self.annotations[~mask]

        self.annotations.set_index('Refseq', inplace=True)

        self.sequences = list(SeqIO.parse(path + "/sequence.fasta", "fasta"))
        self.sequences = [s for s in self.sequences if len(s.seq)]

    def __getitem__(self, idx):
        data = self.sequences[idx]
        seq = str(data.seq)
        label = self.annotations.loc[data.id[:-2]]
        start = label["OriC start"] - 1
        end = label["OriC end"] - 1

        start = start % len(seq)
        end = end % len(seq)

        seq, start, end = random_subsequence(seq, start, end, min(16000, len(seq)))

        return seq, (start, end), str(data.name)

    def __len__(self):
        return len(self.sequences)