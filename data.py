from torch.utils.data import Dataset
from Bio import SeqIO
import pandas as pd
import random
import torch
from sklearn.model_selection import train_test_split

def random_subsequence(sequence, start, end, n):
    """Generate a random subsequence that contains the start and end location"""
    length = len(sequence)
    
    # Step 1: Calculate the effective length of the slice
    if end < start:
        print(end, start)
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

    def __init__(self, annotations, sequences):

        self.annotations = annotations
        self.sequences = list(SeqIO.parse(sequences, "fasta"))

        # Filter out, have to take seq.id[:-2] because the download added .1 or .2 if we had multiple entries
        self.sequences = [seq for seq in self.sequences if seq.id[:-2] in self.annotations.index]
        
        print("Before filtering:", len(self.sequences))
        self.sequences = [seq for seq in self.sequences if len(seq.seq) <= 32000]
        print("After filtering:", len(self.sequences))

    def __getitem__(self, idx):

        data = self.sequences[idx]
        seq = str(data.seq)
        label = self.annotations.loc[data.id[:-2]]
        start = label["OriC start"] - 1
        end = label["OriC end"] - 1

        # In reality, there are just a few instances where start == len(seq). 
        # In this case, this code sets it to 0
        start = start % len(seq)
        end = end % len(seq)    

        #seq, start, end = random_subsequence(seq, start, end, min(16000, len(seq)))

        return seq, (start, end), label.to_dict()

    def __len__(self):
        return len(self.sequences)

def get_split(split_name):

    plasmid_annotations = pd.read_csv("DoriC12.1/DoriC12.1_plasmid.csv")
    print(plasmid_annotations.head())
    mask = plasmid_annotations.duplicated('Refseq', keep=False)
    plasmid_annotations = plasmid_annotations[~mask]
    plasmid_annotations.set_index('Refseq', inplace=True)

    bacteria_annotations = pd.read_csv("DoriC12.1/DoriC12.1_bacteria.csv")
    print(bacteria_annotations.head())
    mask = bacteria_annotations.duplicated('Refseq', keep=False)
    bacteria_annotations = bacteria_annotations[~mask]
    bacteria_annotations.set_index('Refseq', inplace=True)

    if split_name == "plasmid_random":
        train, test = train_test_split(plasmid_annotations, train_size=0.9)
        return OriDataset(train, "/root/autodl-fs/sequences.fasta"), OriDataset(test, "/root/autodl-fs/sequences.fasta")
    if split_name == "bacteria_random":
        train, test = train_test_split(bacteria_annotations, train_size=0.9)
        return OriDataset(train, "~/autodl-fs/bacterial_genomes.fasta"), OriDataset(test, "~/autodl-fs/bacterial_genomes.fasta")
    elif split_name == "taxonomy_islands":

        # Define the taxonomic levels
        levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species', 'Strain']
        
        # Split the taxonomy column
        annotations[levels] = annotations['Lineage'].str.split(',', expand=True)
        
        # Trim whitespace from the new columns
        for level in levels:
            annotations[level] = annotations[level].str.strip()
        
        # Fill NaN values with empty string
        annotations[levels] = annotations[levels].fillna('')

        class_counts = annotations['Class'].value_counts()

        # Create a boolean mask for classes appearing over 25 times
        mask = annotations['Class'].map(class_counts) > 25
        
        # Split the dataframe
        train = annotations[mask]
        test = annotations[~mask]
        
        return OriDataset(train), OriDataset(test)
    else: 
        raise ValueError