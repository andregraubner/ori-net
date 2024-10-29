from torch.utils.data import Dataset
from Bio import SeqIO
import pandas as pd
import random
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

import random

def shift_string_cyclically(s, start, end, max_length):
    # Ensure start and end are valid indices
    if start < 0 or start >= len(s):
        raise ValueError("Start point must be a valid index within the string.")

    # Shift the string so that the start index is at the front
    shifted_string = s[start:] + s[:start]

    # New indices for start and end
    new_start_index = 0
    new_end_index = (end - start) % len(s)

    shift_amount = random.randint(0, min(len(s) - new_end_index - 1, max_length-new_end_index-1))
    shifted_string = shifted_string[-shift_amount:] + shifted_string[:-shift_amount]
    shifted_string = shifted_string[:max_length]
    
    return shifted_string, shift_amount, new_end_index + shift_amount

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

def get_token_index_for_nth_char(input_string, n):
    # Tokenize the input string
    encoded = tokenizer(input_string, return_offsets_mapping=True, add_special_tokens=True)
    
    # Get the offsets mapping
    offsets = encoded['offset_mapping']
    
    # Find the token that contains the n-th character
    for i, (start, end) in enumerate(offsets):
        if start <= n < end:
            return i
    
    # If n is out of range, return None or raise an exception
    return None

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

    def __init__(self, annotations, sequences, max_length):

        self.annotations = annotations
        self.sequences = sequences
        self.max_length = max_length

        # Filter out, have to take seq.id[:-2] because the download added .1 or .2 if we had multiple entries
        print(self.annotations.index[:10])
        print([seq.id for seq in self.sequences][:10])
        #self.sequences = [seq for seq in self.sequences if seq.id[:-2] in self.annotations.index]
        self.sequences = [seq for seq in self.sequences if seq.id in self.annotations.index]
        
        print("Before filtering:", len(self.sequences))
        #self.sequences = [seq for seq in self.sequences if len(seq.seq) <= 100000]
        #self.sequences = [seq for seq in self.sequences if len(seq.seq) <= 50000]
        print("After filtering:", len(self.sequences))

    def __getitem__(self, idx):

        data = self.sequences[idx]
        seq = str(data.seq)
        #label = self.annotations.loc[data.id[:-2]]
        label = self.annotations.loc[data.id]
        if type(label) == pd.DataFrame:
            label = label.sample(1).iloc[0] # we might have multiple Oris, so we sample one.
        start = label["OriC start"] - 1
        end = label["OriC end"] - 1

        # In reality, there are just a few instances where start == len(seq). 
        # In this case, this code sets it to 0
        start = start % len(seq)
        end = end % len(seq)    

        old_slen = len(seq)

        try:
            seq, start, end = shift_string_cyclically(seq, start, end, max_length=self.max_length)
        except:
            seq, start, end = seq[:self.max_length], 0, 1
            
        #start = get_token_index_for_nth_char(seq, start)
        #end = get_token_index_for_nth_char(seq, end)

        #seq = torch.tensor(tokenizer.encode(seq), dtype=torch.long)

        #seq, start, end = random_subsequence(seq, start, end, min(16000, len(seq)))
        
        #print("seq lens:", old_slen, len(seq))
        
        return seq, (start, end), label.to_dict()

    def __len__(self):
        return len(self.sequences)

def get_split(split_name, max_length=None):

    plasmid_annotations = pd.read_csv("DoriC12.1/DoriC12.1_plasmid.csv")
    #mask = plasmid_annotations.duplicated('Refseq', keep=False)
    #plasmid_annotations = plasmid_annotations[~mask]
    plasmid_annotations["NC"] = plasmid_annotations["NC"].astype(str) + ".1"
    plasmid_annotations.set_index('NC', inplace=True)

    bacteria_annotations = pd.read_csv("DoriC12.1/DoriC12.1_bacteria.csv")
    #mask = bacteria_annotations.duplicated('Refseq', keep=False)
    #bacteria_annotations = bacteria_annotations[~mask]
    #bacteria_annotations = bacteria_annotations[bacteria_annotations["Assembly level"] == "Complete"]
    bacteria_annotations.set_index('NC', inplace=True)

    if split_name == "plasmid_random":
        train, test = train_test_split(plasmid_annotations, train_size=0.9)
        sequences = list(SeqIO.parse("/root/autodl-fs/sequences.fasta", "fasta"))
        return OriDataset(train, sequences, max_length), OriDataset(test, sequences, max_length)
    if split_name == "bacteria_random":
        train, test = train_test_split(bacteria_annotations, train_size=0.999)
        sequences = list(SeqIO.parse("/root/autodl-fs/bacterial_genomes.fasta", "fasta"))
        return OriDataset(train, sequences), OriDataset(test, sequences)
    elif split_name == "bacteria_taxonomy_islands":

        # Define the taxonomic levels
        levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species', 'Strain']
        
        # Split the taxonomy column
        print(bacteria_annotations['Lineage'])
        
        #bacteria_annotations[levels] = bacteria_annotations['Lineage'].str.split(',', expand=True)
        bacteria_annotations["lens"] = bacteria_annotations['Lineage'].str.split(',').str.len()
        
        print(bacteria_annotations["lens"].describe())
        print(bacteria_annotations.sort_values('lens'))
        quit()
        
        # Trim whitespace from the new columns
        for level in levels:
            bacteria_annotations[level] = bacteria_annotations[level].str.strip()
        
        # Fill NaN values with empty string
        bacteria_annotations[levels] = bacteria_annotations[levels].fillna('')

        class_counts = bacteria_annotations['Class'].value_counts()

        # Create a boolean mask for classes appearing over 25 times
        mask = bacteria_annotations['Class'].map(class_counts) > 25
        
        # Split the dataframe
        train = plasmid_annotations[mask]
        test = plasmid_annotations[~mask]

        print(len(train))
        print(len(test))
        quit()
    
    elif split_name == "plasmid_taxonomy_islands":

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
    else: 
        raise ValueError