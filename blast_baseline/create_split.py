from Bio import SeqIO
import pandas as pd
import random

plasmid_annotations = pd.read_csv("DoriC12.1/DoriC12.1_plasmid.csv")
plasmid_annotations["NC"] = plasmid_annotations["NC"].astype(str) + ".1"
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

# Create fasta file containing only ori sequences for blast DB
print(f"Creating ori database of {len(train)} oris")
with open('oris.fasta', 'w') as f:
    for k, v in train["OriC sequence"].items():
        f.write(f'>{k}\n{v}\n')


sequences = {seq.id: str(seq.seq) for seq in SeqIO.parse("sequences.fasta", "fasta") if seq.id in test.index}
print(f"Creating test database of {len(sequences)} sequences")
with open('test.fasta', 'w') as f:
    for k, v in sequences.items():
        f.write(f'>{k}\n{v}\n') 
