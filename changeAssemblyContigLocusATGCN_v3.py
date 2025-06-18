from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import re
import sys

if len(sys.argv) != 2:
    print("Usage: need a assembly file")
    exit()

assembly = sys.argv[1]

i = 0
for record in SeqIO.parse(assembly, format="fasta"):
    i = i + 1
    new_id = assembly[0:14] + 'c' + str(i)
    sequence=re.sub("[^ATGCN]", "N", str(record.seq))
    new_rec=SeqRecord(id=new_id, seq=Seq(sequence), description='')
    SeqIO.write(new_rec, sys.stdout, 'fasta')