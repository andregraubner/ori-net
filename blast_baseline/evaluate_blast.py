from Bio import SeqIO
import pandas as pd
import random
from sklearn.metrics import jaccard_score
import numpy as np

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
test = plasmid_annotations[~mask]

# Load results
results = pd.read_csv("results.csv", names=["id", "evalue", "length", "start", "end"])
results.set_index("id", inplace=True)

sequences = {seq.id: str(seq.seq) for seq in SeqIO.parse("sequences.fasta", "fasta") if seq.id in test.index}

ious = []

for name, row in test.iterrows():
    if row["Refseq"] + ".1" in results.index:
       
        pred = results.loc[row["Refseq"] + ".1"]
 
        if isinstance(pred, pd.Series):
            pred = pred  # Single row, already a Series
        else:  # DataFrame with multiple rows
            #pred = pred.sort_values('length', ascending=False)
            pred = pred.iloc[0]  # Get first row as Series

        seq = sequences[row["Refseq"] + ".1"]

        preds = np.zeros(len(seq))
        labels = np.zeros(len(seq))
        start, end = row["OriC start"], row["OriC end"]        

        if start < end:
            labels[start:end] = 1
        else:
            labels[:end] = 1
            labels[start:] = 1

        print(int(pred["end"]) - int(pred["start"]))

        preds[int(pred["start"]):int(pred["end"])] = 1

        iou = jaccard_score(labels, preds)
        ious.append(iou)
    
    else:
        ious.append(0)

print(ious)
print(np.mean(ious))
