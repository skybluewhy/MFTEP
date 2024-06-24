import numpy as np
import os
from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA
import torch
import tools.gvp.data as gvp_data
from Bio.PDB import PDBParser
from get_graph import smile_to_graph
from pytoda.proteins import aas_to_smiles


blosum62 = {
    "A": np.array(
        (4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0)
    ),
    "R": np.array(
        (-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3)
    ),
    "N": np.array(
        (-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3)
    ),
    "D": np.array(
        (-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3)
    ),
    "C": np.array(
        ( 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1)
    ),
    "Q": np.array(
        (-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2)
    ),
    "E": np.array(
        (-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2)
    ),
    "G": np.array(
        ( 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3)
    ),
    "H": np.array(
        (-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3)
    ),
    "I": np.array(
        (-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3, 1, 0, -3, -2, -1, -3, -1,  3)
    ),
    "L": np.array(
        (-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1,  1)
    ),
    "K": np.array(
        (-1,  2, 0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1, 0, -1, -3, -2, -2)
    ),
    "M": np.array(
        (-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1)
    ),
    "F": np.array(
        (-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1)
    ),
    "P": np.array(
        (-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2)
    ),
    "S": np.array(
        (1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2)
    ),
    "T": np.array(
        (0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0)
    ),
    "W": np.array(
        (-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11,  2, -3)
    ),
    "Y": np.array(
        (-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1)
    ),
    "V": np.array(
        (0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4)
    ),
}

alphabet_num = {
    "A": 1,
    "R": 2,
    "N": 3,
    "D": 4,
    "C": 5,
    "Q": 6,
    "E": 7,
    "G": 8,
    "H": 9,
    "I": 10,
    "L": 11,
    "K": 12,
    "M": 13,
    "F": 14,
    "P": 15,
    "S": 16,
    "T": 17,
    "W": 18,
    "Y": 19,
    "V": 20,
}

three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
                'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}



def get_clean_res_list(res_list, verbose=False, ensure_ca_exist=False, bfactor_cutoff=None):
    clean_res_list = []
    cnt = 0
    for res in res_list:
        cnt += 1
        hetero, resid, insertion = res.full_id[-1]
        if hetero == ' ':
            if res.resname not in three_to_one:
                if verbose:
                    print(res, "has non-standard resname")
                continue
            if (not ensure_ca_exist) or ('CA' in res):
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res['CA'].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)
        else:
            if verbose:
                print(res, res.full_id, "is hetero")
    return clean_res_list


def get_gvp_feat(seq, top_k=5):
    pdb = seq
    proteinFile = "./data/TITAN_pdb/" + seq + ".pdb"
    parser = PDBParser(QUIET=True)
    s = parser.get_structure(pdb, proteinFile)  # read protein PDB file
    res_list = get_clean_res_list(s.get_residues(), verbose=False, ensure_ca_exist=True)  # obtain valid residues
    # protein feature extraction code from https://github.com/drorlab/gvp-pytorch
    # ensure all res contains N, CA, C and O
    res_list = [res for res in res_list if (('N' in res) and ('CA' in res) and ('C' in res) and ('O' in res))]
    # construct the input for ProteinGraphDataset
    # which requires name, seq, and a list of shape N * 4 * 3
    structure = {}
    structure['name'] = "placeholder"
    structure['seq'] = "".join([three_to_one.get(res.resname) for res in res_list])
    coords = []
    for res in res_list:
        res_coords = []
        for atom in [res['N'], res['CA'], res['C'], res['O']]:
            res_coords.append(list(atom.coord))
        coords.append(res_coords)
    structure['coords'] = coords
    torch.set_num_threads(1)  # this reduce the overhead, and speed up the process for me.
    dataset = gvp_data.ProteinGraphDataset([structure, structure], top_k=top_k)
    return dataset[0]


class DTADataset(InMemoryDataset):
    def __init__(self, root='data', dataset='PP3',
                 xd=None, y=None, transform=None, pre_transform=None,
                 all_CDR3_blosum62_feat=None, all_epitope_blosum62_feat=None,
                 all_CDR3_seq=None, all_epitope_seq=None,
                 all_CDR3_seq_len=None, all_epitope_seq_len=None,
                 all_CDR3_molecule_map=None, all_epitope_molecule_map=None,
                 all_CDR3_molecule_len=None, all_epitope_molecule_len=None,
                 all_labels=None):

        super(DTADataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.all_CDR3_blosum62_feat = all_CDR3_blosum62_feat
        self.all_epitope_blosum62_feat = all_epitope_blosum62_feat
        self.all_CDR3_seq = all_CDR3_seq
        self.all_epitope_seq = all_epitope_seq
        self.all_CDR3_seq_len = all_CDR3_seq_len
        self.all_epitope_seq_len = all_epitope_seq_len
        self.all_CDR3_molecule_len = all_CDR3_molecule_len
        self.all_epitope_molecule_len = all_epitope_molecule_len
        self.process(all_CDR3_molecule_map, all_epitope_molecule_map, all_labels)

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return None

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, all_CDR3_molecule_map, all_epitope_molecule_map, labels):
        data_list_CDR3 = []
        data_list_CDR3_gvp = []
        data_list_epitope = []
        data_list_epitope_gvp = []
        data_len = len(all_CDR3_molecule_map)
        for i in range(data_len):
            CDR3_size, CDR3_features, CDR3_edge_index = all_CDR3_molecule_map[i]
            GCNData_CDR3 = DATA.Data(x=torch.Tensor(CDR3_features),
                                    edge_index=torch.LongTensor(CDR3_edge_index).transpose(1, 0),
                                    y=torch.LongTensor([labels[i]]))

            GCNData_CDR3.all_CDR3_blosum62_feat = torch.tensor([self.all_CDR3_blosum62_feat[i]], dtype=torch.float32)
            GCNData_CDR3.all_epitope_blosum62_feat = torch.tensor([self.all_epitope_blosum62_feat[i]], dtype=torch.float32)

            GCNData_CDR3.all_CDR3_seq_len = torch.tensor(self.all_CDR3_seq_len[i], dtype=torch.long)
            GCNData_CDR3.all_epitope_seq_len = torch.tensor(self.all_epitope_seq_len[i], dtype=torch.long)
            GCNData_CDR3.all_CDR3_molecule_len = torch.tensor(self.all_CDR3_molecule_len[i], dtype=torch.long)
            GCNData_CDR3.all_epitope_molecule_len = torch.tensor(self.all_epitope_molecule_len[i], dtype=torch.long)

            GCNData_CDR3.__setitem__('c_size', torch.LongTensor([CDR3_size]))

            epitope_size, epitope_features, epitope_edge_index = all_epitope_molecule_map[i]
            GCNData_epitope = DATA.Data(x=torch.Tensor(epitope_features),
                                    edge_index=torch.LongTensor(epitope_edge_index).transpose(1, 0),
                                    y=torch.LongTensor([labels[i]]))
            GCNData_epitope.__setitem__('epitope_size', torch.LongTensor([epitope_size]))

            CDR3_gvp_feat = get_gvp_feat(self.all_CDR3_seq[i], top_k=5)
            CDR3_gvp_feat.__setitem__('CDR3_gvp_size', torch.LongTensor([CDR3_gvp_feat.x.shape[0]]))

            epitope_gvp_feat = get_gvp_feat(self.all_epitope_seq[i], top_k=2)
            epitope_gvp_feat.__setitem__('epitope_gvp_size', torch.LongTensor([epitope_gvp_feat.x.shape[0]]))

            data_list_CDR3.append(GCNData_CDR3)
            data_list_epitope.append(GCNData_epitope)

            data_list_CDR3_gvp.append(CDR3_gvp_feat)
            data_list_epitope_gvp.append(epitope_gvp_feat)

        if self.pre_filter is not None:
            data_list_CDR3 = [data for data in data_list_CDR3 if self.pre_filter(data)]
            data_list_epitope = [data for data in data_list_epitope if self.pre_filter(data)]

            data_list_CDR3_gvp = [data for data in data_list_CDR3_gvp if self.pre_filter(data)]
            data_list_epitope_gvp = [data for data in data_list_epitope_gvp if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list_CDR3 = [self.pre_transform(data) for data in data_list_CDR3]
            data_list_epitope = [self.pre_transform(data) for data in data_list_epitope]

            data_list_CDR3_gvp = [self.pre_transform(data) for data in data_list_CDR3_gvp]
            data_list_epitope_gvp = [self.pre_transform(data) for data in data_list_epitope_gvp]

        self.data_CDR3 = data_list_CDR3
        self.data_epitope = data_list_epitope
        self.data_CDR3_gvp = data_list_CDR3_gvp
        self.data_epitope_gvp = data_list_epitope_gvp

    def __len__(self):
        return len(self.data_CDR3)

    def __getitem__(self, idx):
        return self.data_CDR3[idx], self.data_epitope[idx], self.data_CDR3_gvp[idx],  self.data_epitope_gvp[idx]


def get_dataset(filename):
    all_CDR3_seq = []
    all_epitope_seq = []
    all_CDR3_blosum62_feat = []
    all_CDR3_seq_len = []
    all_CDR3_molecule_len = []
    all_epitope_blosum62_feat = []
    all_epitope_seq_len = []
    all_epitope_molecule_len = []
    all_labels = []
    all_CDR3_molecule_map = []
    all_epitope_molecule_map = []

    index = 0
    f = open(filename, 'r')
    for line in f:
        if index == 0:
            index = 1
            continue
        index += 1
        line = line.replace("\n", "")
        line_vec = line.split(",")
        CDR3_blosum62_feat = []
        epitope_blosum62_feat = []
        CDR3, epitope, label = line_vec[0], line_vec[1], line_vec[2]

        for k in range(35):
            if k < len(CDR3):
                CDR3_blosum62_feat.append(list(blosum62[CDR3[k]]))
            else:
                CDR3_blosum62_feat.append([0 for j in range(20)])
        for k in range(22):
            if k < len(epitope):
                epitope_blosum62_feat.append(list(blosum62[epitope[k]]))
            else:
                epitope_blosum62_feat.append([0 for j in range(20)])
        all_CDR3_blosum62_feat.append(CDR3_blosum62_feat)
        all_CDR3_seq.append(CDR3)
        all_CDR3_seq_len.append(len(CDR3))
        CDR3_g = smile_to_graph(aas_to_smiles(CDR3))
        all_CDR3_molecule_len.append(CDR3_g[0])
        all_CDR3_molecule_map.append(CDR3_g)

        all_epitope_blosum62_feat.append(epitope_blosum62_feat)
        all_epitope_seq.append(epitope)
        all_epitope_seq_len.append(len(epitope))
        epitope_g = smile_to_graph(aas_to_smiles(epitope))
        all_epitope_molecule_len.append(epitope_g[0])
        all_epitope_molecule_map.append(epitope_g)

        all_labels.append(float(label))

    data_set = DTADataset(root='data', dataset='tcr-epitope', all_CDR3_blosum62_feat=all_CDR3_blosum62_feat, all_epitope_blosum62_feat=all_epitope_blosum62_feat,
                          all_CDR3_seq=all_CDR3_seq, all_epitope_seq=all_epitope_seq,
                          all_CDR3_seq_len=all_CDR3_seq_len, all_epitope_seq_len=all_epitope_seq_len,
                          all_CDR3_molecule_map=all_CDR3_molecule_map, all_epitope_molecule_map=all_epitope_molecule_map,
                          all_CDR3_molecule_len=all_CDR3_molecule_len, all_epitope_molecule_len=all_epitope_molecule_len,
                          all_labels=all_labels)
    return data_set


def collate(data_list):
    CDR3_contact_map_batch = Batch.from_data_list([data[0] for data in data_list])
    epitope_contact_map_batch = Batch.from_data_list([data[1] for data in data_list])
    CDR3_gvp_batch = Batch.from_data_list([data[2] for data in data_list])
    epitope_gvp_batch = Batch.from_data_list([data[3] for data in data_list])
    return CDR3_contact_map_batch, epitope_contact_map_batch, CDR3_gvp_batch, epitope_gvp_batch
