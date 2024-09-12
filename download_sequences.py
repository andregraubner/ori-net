from Bio import Entrez, SeqIO
import pandas as pd
from tqdm import tqdm

annotations = pd.read_csv("./DoriC12.1/DoriC12.1_plasmid.csv")

# Provide your email address (required by NCBI)
Entrez.email = "andre.graubner@web.com"

# List of accession numbers
accession_list = list(annotations["Refseq"])

# Output file name
output_file = "sequences_bulk.fasta"

"""
# Fetch sequences and write to FASTA file
with open(output_file, "w") as out_handle:
    for accession in tqdm(accession_list):
        # Fetch the sequence from NCBI
        try:
            handle = Entrez.efetch(db="nucleotide", id=accession, rettype="fasta", retmode="text")
            
            # Read the sequence and write to the output file
            record = SeqIO.read(handle, "fasta")
            SeqIO.write(record, out_handle, "fasta")
            
            handle.close()
        except:
            print("Issue downloading", accession)
"""
handle = Entrez.efetch(db="nucleotide", id=",".join(accession_list[:100]), rettype="fasta", retmode="text")
            
out_handle = open("saved.fasta", "w")
for line in handle:
    out_handle.write(line)
out_handle.close()
handle.close()

print(f"Sequences have been downloaded and saved to {output_file}")