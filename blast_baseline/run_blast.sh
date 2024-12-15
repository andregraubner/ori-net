#!/bin/bash

# Create BLAST database from oris.fasta
makeblastdb -in oris.fasta -dbtype nucl -out oris_db

blastn -query test_blast.fasta -db oris_db -outfmt "10 qseqid evalue length qstart qend" -out "results.csv"
