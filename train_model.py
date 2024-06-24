import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from data import get_dataset, collate
from model import multimodal_model
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from functions import DiffLoss, MSE, CMD


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def make_dir(fp):
    if not os.path.exists(fp):
        os.makedirs(fp, exist_ok=True)


def get_cmd_loss(model):
    loss_cmd = CMD()
    loss = loss_cmd(model.utt_shared_gvp, model.utt_shared_graph, 5)
    loss += loss_cmd(model.utt_shared_gvp, model.utt_shared_blosum, 5)
    loss += loss_cmd(model.utt_shared_graph, model.utt_shared_blosum, 5)
    loss = loss / 3.0
    return loss


def get_diff_loss(model):
    loss_diff = DiffLoss()

    shared_gvp = model.utt_shared_gvp
    shared_graph = model.utt_shared_graph
    shared_blosum = model.utt_shared_blosum
    private_gvp = model.utt_private_gvp
    private_graph = model.utt_private_graph
    private_blosum = model.utt_private_blosum

    # Between private and shared
    loss = loss_diff(private_gvp, shared_gvp)
    loss += loss_diff(private_graph, shared_graph)
    loss += loss_diff(private_blosum, shared_blosum)

    # Across privates
    loss += loss_diff(private_gvp, private_graph)
    loss += loss_diff(private_gvp, private_blosum)
    loss += loss_diff(private_graph, private_blosum)

    return loss


def get_recon_loss(model):
    loss_recon = MSE()
    loss = loss_recon(model.utt_gvp_recon, model.utt_gvp_orig)
    loss += loss_recon(model.utt_graph_recon, model.utt_graph_orig)
    loss += loss_recon(model.utt_blosum_recon, model.utt_blosum_orig)
    loss = loss / 3.0
    return loss


def train_multimodal(model, trainloader, optimizer, device):
    criterion = nn.CrossEntropyLoss(size_average=False)
    model.train()
    running_loss = 0.0
    total = 0
    with torch.autograd.set_detect_anomaly(True):
        for i, data in enumerate(trainloader):
            optimizer.zero_grad()
            CDR3_contact_map_batch, epitope_contact_map_batch = data[0].to(device), data[1].to(device)
            CDR3_gvp_batch = data[2].to(device)
            CDR3_gvp_nodes = (CDR3_gvp_batch.node_s, CDR3_gvp_batch.node_v)
            CDR3_gvp_edges = (CDR3_gvp_batch.edge_s, CDR3_gvp_batch.edge_v)
            CDR3_gvp_edges_index = CDR3_gvp_batch.edge_index
            CDR3_gvp_batch_batch = CDR3_gvp_batch.batch

            epitope_gvp_batch = data[3].to(device)
            epitope_gvp_nodes = (epitope_gvp_batch.node_s, epitope_gvp_batch.node_v)
            epitope_gvp_edges = (epitope_gvp_batch.edge_s, epitope_gvp_batch.edge_v)
            epitope_gvp_edges_index = epitope_gvp_batch.edge_index
            epitope_gvp_batch_batch = epitope_gvp_batch.batch

            labels = CDR3_contact_map_batch.y
            CDR3_blosum62_feat_batch, epitope_blosum62_feat_batch = CDR3_contact_map_batch.all_CDR3_blosum62_feat, CDR3_contact_map_batch.all_epitope_blosum62_feat
            CDR3_seq_len_batch, epitope_seq_len_batch = CDR3_contact_map_batch.all_CDR3_seq_len, CDR3_contact_map_batch.all_epitope_seq_len
            CDR3_molecule_len_batch, epitope_molecule_len_batch = CDR3_contact_map_batch.all_CDR3_molecule_len, CDR3_contact_map_batch.all_epitope_molecule_len
            outputs = model(CDR3_contact_map_batch, epitope_contact_map_batch,
                            CDR3_blosum62_feat_batch, epitope_blosum62_feat_batch,
                            CDR3_gvp_nodes, CDR3_gvp_edges, CDR3_gvp_edges_index, CDR3_gvp_batch_batch,
                            epitope_gvp_nodes, epitope_gvp_edges, epitope_gvp_edges_index, epitope_gvp_batch_batch,
                            CDR3_seq_len_batch, epitope_seq_len_batch,
                            CDR3_molecule_len_batch, epitope_molecule_len_batch,
                            device)
            diff_loss = get_diff_loss(model.CDR3_fusion) + get_diff_loss(model.epitope_fusion)
            recon_loss = get_recon_loss(model.CDR3_fusion) + get_recon_loss(model.epitope_fusion)
            cmd_loss = get_cmd_loss(model.CDR3_fusion) + get_cmd_loss(model.epitope_fusion)
            loss = criterion(outputs, labels) + 0.3 * diff_loss + 1 * cmd_loss + 1 * recon_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.shape[0]
            total = total + labels.shape[0]


def test_multimodal(model, testloader, best_auc, device, epoch, repeat, dataset, is_train=True):
    model.eval()
    pred_val = []
    y_true_s = []
    with torch.no_grad():
        for i, data in enumerate(testloader):
            CDR3_contact_map_batch, epitope_contact_map_batch = data[0].to(device), data[1].to(device)
            CDR3_gvp_batch = data[2].to(device)
            CDR3_gvp_nodes = (CDR3_gvp_batch.node_s, CDR3_gvp_batch.node_v)
            CDR3_gvp_edges = (CDR3_gvp_batch.edge_s, CDR3_gvp_batch.edge_v)
            CDR3_gvp_edges_index = CDR3_gvp_batch.edge_index
            CDR3_gvp_batch_batch = CDR3_gvp_batch.batch

            epitope_gvp_batch = data[3].to(device)
            epitope_gvp_nodes = (epitope_gvp_batch.node_s, epitope_gvp_batch.node_v)
            epitope_gvp_edges = (epitope_gvp_batch.edge_s, epitope_gvp_batch.edge_v)
            epitope_gvp_edges_index = epitope_gvp_batch.edge_index
            epitope_gvp_batch_batch = epitope_gvp_batch.batch

            labels = CDR3_contact_map_batch.y
            CDR3_blosum62_feat_batch, epitope_blosum62_feat_batch = CDR3_contact_map_batch.all_CDR3_blosum62_feat, CDR3_contact_map_batch.all_epitope_blosum62_feat
            CDR3_seq_len_batch, epitope_seq_len_batch = CDR3_contact_map_batch.all_CDR3_seq_len, CDR3_contact_map_batch.all_epitope_seq_len
            CDR3_molecule_len_batch, epitope_molecule_len_batch = CDR3_contact_map_batch.all_CDR3_molecule_len, CDR3_contact_map_batch.all_epitope_molecule_len
            outputs = model(CDR3_contact_map_batch, epitope_contact_map_batch,
                            CDR3_blosum62_feat_batch, epitope_blosum62_feat_batch,
                            CDR3_gvp_nodes, CDR3_gvp_edges, CDR3_gvp_edges_index, CDR3_gvp_batch_batch,
                            epitope_gvp_nodes, epitope_gvp_edges, epitope_gvp_edges_index, epitope_gvp_batch_batch,
                            CDR3_seq_len_batch, epitope_seq_len_batch,
                            CDR3_molecule_len_batch, epitope_molecule_len_batch,
                            device)

            p = F.softmax(outputs, dim=1)[:, 1]
            y_true_s = y_true_s + labels.tolist()
            pred_val = pred_val + p.tolist()
        AUC = roc_auc_score(y_true_s, pred_val)
        if AUC > best_auc and is_train:
            torch.save(model.state_dict(),
                       './model/' + str(dataset) + '/repeat' + str(repeat) + '/checkpoint.pt')
        best_auc = max(best_auc, AUC)
        print("Epoch: " + str(epoch) + " AUC: " + str(AUC) + " best AUC: " + str(best_auc))
    return best_auc, AUC


def run_multimodal_model(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_dataset = args.train_dataset
    test_dataset = args.test_dataset
    train_dataset = get_dataset(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               collate_fn=collate)
    test_dataset = get_dataset(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              collate_fn=collate)
    model = multimodal_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                           weight_decay=args.weight_decay, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=6, verbose=False,
                                                           threshold=0.00005, eps=1e-08)
    best_auc = 0
    for epoch in range(args.epoch):
        train_multimodal(model, train_loader, optimizer, device)
        best_auc, auc_res = test_multimodal(model, test_loader, best_auc, device, epoch, args.rp, args.dataset)
        scheduler.step(auc_res)


def test_multimodal_model(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    test_dataset = args.test_dataset
    test_dataset = get_dataset(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              collate_fn=collate)
    model = multimodal_model().to(device)
    model.load_state_dict(torch.load(args.save_model, map_location=device))
    best_auc = 0
    best_auc, auc_res = test_multimodal(model, test_loader, best_auc, device, 0, args.rp, args.dataset, False)
    print("AUC: " + str(best_auc) + "\n\n\n")


def parser():
    ap = argparse.ArgumentParser(description='TCR-peptide model')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--epoch', type=int, default=50, help='Number of epochs. Default is 100.')
    ap.add_argument('--batch_size', type=int, default=64, help='Batch size. Default is 8.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--lr', default=0.001)
    ap.add_argument('--weight_decay', default=1e-4)
    ap.add_argument('--num_workers', default=0, type=int)
    ap.add_argument('--save_dir_format', default='./results_dpp/{}/repeat{}/')
    ap.add_argument('--dataset', default='strict_split', type=str)
    ap.add_argument('--train_dataset', default='./data/strict_split/fold0/train.csv', type=str)
    ap.add_argument('--test_dataset', default='./data/strict_split/fold0/test.csv', type=str)
    ap.add_argument('--only_test', default=False, type=bool)
    ap.add_argument('--save_model', default="./checkpoint.pt", type=str)
    args = ap.parse_args()
    return args


if __name__ == '__main__':
    args = parser()
    for rp in range(args.repeat):
        print('This is repeat ', rp)
        args.rp = rp
        args.save_dir = args.save_dir_format.format(args.dataset, args.rp)
        make_dir(args.save_dir)
        make_dir('./model/' + str(args.dataset) + '/repeat' + str(args.rp))
        sys.stdout = Logger(args.save_dir + 'log.txt')
        if args.only_test:
            test_multimodal_model(args)
        else:
            run_multimodal_model(args)
