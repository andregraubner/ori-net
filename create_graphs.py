from Bio import SeqIO
import pandas as pd
import random
from sklearn.metrics import jaccard_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
test = plasmid_annotations[~mask]

# Load results
sequences = {seq.id: str(seq.seq) for seq in SeqIO.parse("/root/autodl-fs/sequences.fasta", "fasta")}

orint_results = np.load("orint_out.npy", allow_pickle=True)[()]
caduceus_results = np.load("caduceus_out.npy", allow_pickle=True)[()]
blast_results = pd.read_csv("blast_baseline/results.csv", names=["id", "evalue", "length", "start", "end"])
blast_results.set_index("id", inplace=True)

orint_ious = []
caduceus_ious = []
blast_ious = []

orint_probs = []
caduceus_probs = []
all_labels = []

for name, row in test.iterrows():

    if row["Refseq"] + ".1" not in sequences:
        # Why could this be happening? TODO: Investigate...
        continue
    
    seq = sequences[row["Refseq"] + ".1"]
    labels = np.zeros(len(seq))
    start, end = row["OriC start"], row["OriC end"]        

    if start < end:
        labels[start:end] = 1
    else:
        labels[:end] = 1
        labels[start:] = 1

    all_labels.append(labels)
        
    if row["Refseq"] + ".1" in orint_results.keys(): 
        orint_preds = orint_results[row["Refseq"] + ".1"]
        orint_preds = np.nan_to_num(orint_preds) # why is there sometimes a single nan here? TODO: investigate
        orint_ious.append(jaccard_score(labels, orint_preds > 0.5))
        orint_probs.append(orint_preds)
    else:
        orint_results.append(0)
        orint_probs.append(np.zeros_like(labels))
        
    if row["Refseq"] + ".1" in caduceus_results.keys():
        caduceus_preds = caduceus_results[row["Refseq"] + ".1"]
        caduceus_ious.append(jaccard_score(labels, caduceus_preds > 0.5))
        caduceus_probs.append(caduceus_preds)
    else:
        caduceus_results.append(0)
        caduceus_probs.append(np.zeros_like(labels))

    if row["Refseq"] + ".1" in blast_results.index:
        pred = blast_results.loc[row["Refseq"] + ".1"]

        if isinstance(pred, pd.Series):
            pred = pred  # Single row, already a Series
        else:  # DataFrame with multiple rows
            #pred = pred.sort_values('length', ascending=False)
            pred = pred.iloc[0]  # Get first row as Series
        
        preds = np.zeros(len(seq))
        start, end = row["OriC start"], row["OriC end"]       
        pred_start = int(pred["start"]) % len(seq)
        pred_end = int(pred["end"]) % len(seq)
        preds[pred_start:pred_end] = 1
        blast_ious.append(jaccard_score(labels, preds))
    else:
        blast_ious.append(0)

# Create bar plot

print(np.mean(orint_ious), np.mean(caduceus_ious), np.mean(blast_ious))

# Create figure with two subplots
fig, ax1 = plt.subplots(1, 1, figsize=(4, 5))

# Plot 1: Boxplot
sns.barplot(data=[orint_ious, caduceus_ious, blast_ious], 
            width=0.5,
            palette="Set2",
            ax=ax1)
ax1.set_xticklabels(['Ori-NT', 'Ori-Caduceus', 'Blast'])
ax1.set_title('Mean IoU Scores')
ax1.set_ylabel('IoU Score')

plt.ylim(0,1)

plt.tight_layout()
plt.show()
plt.savefig("plot.png", dpi=300)


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

targets = np.concatenate(all_labels)
preds_nt = np.concatenate(orint_probs)
preds_caduceus = np.concatenate(caduceus_probs)

# Calculate precision-recall curves for each model
precision_nt, recall_nt, _ = precision_recall_curve(targets, preds_nt)
precision_cad, recall_cad, _ = precision_recall_curve(targets, preds_caduceus)

# Calculate average precision scores
ap_nt = average_precision_score(targets, preds_nt)
ap_cad = average_precision_score(targets, preds_caduceus)

# Set style
sns.set_palette("husl")

# Create the plot
plt.figure(figsize=(4, 5))

# Plot the precision-recall curves
plt.plot(recall_nt, precision_nt, lw=2, label=f'NT (AP = {ap_nt:.3f})')
plt.plot(recall_cad, precision_cad, lw=2, label=f'Caduceus (AP = {ap_cad:.3f})')

# Customize the plot
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves', fontsize=14, pad=20)
plt.legend(loc='best', fontsize=10)

# Set axis limits
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Tight layout
plt.tight_layout()

plt.show()
plt.savefig("precision_recall.png", dpi=300)