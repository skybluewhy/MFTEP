import torch
import esm
import biotite.structure.io as bsio
import time


model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()


def get_pdb(sequence, filename):
    with torch.no_grad():
        output = model.infer_pdb(sequence)

    with open(filename, "w") as f:
        f.write(output)

    struct = bsio.load_structure(filename, extra_fields=["b_factor"])
    print(struct.b_factor.mean())


all_seqs = []
f = open("all_seqs.txt", 'r')
for line in f:
    line = line.replace("\n", "")
    all_seqs.append(line)
f.close()

for line in all_seqs:
    start = time.time()
    get_pdb(line, "./TITAN_pdb/" + line + ".pdb")
    end = time.time()
    print("time: " + str(end - start))
