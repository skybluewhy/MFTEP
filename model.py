import torch.nn as nn
import torch
from tools.gvp import GVP, GVPConvLayer, LayerNorm
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch.autograd import Function


def get_nodes_att_f(x, batch_len, max_len=254, device="cpu", dim=64):
    x_nodes = torch.split(x, batch_len.tolist(), dim=0)
    x_nodes_pad = [torch.tensor([[0 for i in range(dim)] for j in range(max_len - k)]).to(device) for k in batch_len.tolist()]
    x_nodes = torch.cat(
        [torch.cat([x_nodes[i], x_nodes_pad[i]], dim=0).unsqueeze(dim=0) for i in range(len(x_nodes))], dim=0)
    return x_nodes


class gvp_emb(nn.Module):
    '''
    GVP-GNN for Model Quality Assessment as described in manuscript.

    Takes in protein structure graphs of type `torch_geometric.data.Data`
    or `torch_geometric.data.Batch` and returns a scalar score for
    each graph in the batch in a `torch.Tensor` of shape [n_nodes]

    Should be used with `gvp.data.ProteinGraphDataset`, or with generators
    of `torch_geometric.data.Batch` objects with the same attributes.

    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param node_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP-GNN layers
    :seq_in: if `True`, sequences will also be passed in with
             the forward pass; otherwise, sequence information
             is assumed to be part of input node embeddings
    :param num_layers: number of GVP-GNN layers
    :param drop_rate: rate to use in all dropout layers
    '''

    def __init__(self, node_in_dim, node_h_dim,
                 edge_in_dim, edge_h_dim,
                 num_layers=3, drop_rate=0.1, seq_len=22):

        super(gvp_emb, self).__init__()

        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )

        self.layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers))

        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))

        self.dense = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64)
        )

    def forward(self, h_V, edge_index, h_E, batch, batch_len, max_len, device):
        '''
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        '''
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)
        # out = scatter_mean(out, batch, dim=0)
        out_pad = get_nodes_att_f(out, batch_len, max_len=max_len, device=device, dim=6)
        out_pad = self.dense(out_pad)
        out = torch.sum(out_pad, dim=1)
        return out_pad, out


class GINConvNet(torch.nn.Module):
    def __init__(self, dim=64, num_features_xd=78):

        super(GINConvNet, self).__init__()

        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.norm = nn.LayerNorm(64)
        self.fc1_xd = Linear(dim, dim)

    def forward(self, batch_nodes_i, batch_edge_index, batch_batch, batch_len, max_len, device):
        x, edge_index, batch = batch_nodes_i, batch_edge_index, batch_batch

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)

        x_seq = get_nodes_att_f(x, batch_len, max_len=max_len, device=device)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.norm(x)
        return x_seq, x


class Mlp_layer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(hidden_features) or in_features
        self.norm = norm_layer(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class graph_unimodal_model(nn.Module):
    def __init__(self):
        super(graph_unimodal_model, self).__init__()
        self.CDR3_graph_enc = GINConvNet()
        self.epitope_graph_enc = GINConvNet()

    def forward(self, CDR3_graph, epitope_graph,
                CDR3_len_batch, epitope_len_batch, device):
        batch_nodes_CDR3, batch_edge_index_CDR3, batch_batch_CDR3 = CDR3_graph.x, CDR3_graph.edge_index, CDR3_graph.batch
        CDR3_g_seq, CDR3_g_feat = self.CDR3_graph_enc(batch_nodes_CDR3, batch_edge_index_CDR3, batch_batch_CDR3, CDR3_len_batch, 254, device)

        batch_nodes_epitope, batch_edge_index_epitope, batch_batch_epitope = epitope_graph.x, epitope_graph.edge_index, epitope_graph.batch
        epitope_g_seq, epitope_g_feat = self.epitope_graph_enc(batch_nodes_epitope, batch_edge_index_epitope, batch_batch_epitope, epitope_len_batch, 179, device)
        return CDR3_g_seq, CDR3_g_feat, epitope_g_seq, epitope_g_feat


class gvp_unimodal_model(nn.Module):
    def __init__(self):
        super(gvp_unimodal_model, self).__init__()
        node_dim = (6, 3)
        edge_dim = (32, 1)
        self.CDR3_gvp_layer = gvp_emb(node_dim, node_dim, edge_dim, edge_dim, seq_len=35)
        self.epitope_gvp_layer = gvp_emb(node_dim, node_dim, edge_dim, edge_dim, seq_len=22)

    def forward(self, CDR3_gvp_nodes, CDR3_gvp_edges, CDR3_gvp_edges_index,
                epitope_gvp_nodes, epitope_gvp_edges, epitope_gvp_edges_index,
                CDR3_gvp_batch_batch, epitope_gvp_batch_batch,
                CDR3_len_batch, epitope_len_batch, device):
        CDR3_gvp_seq, CDR3_gvp_feat = self.CDR3_gvp_layer(CDR3_gvp_nodes, CDR3_gvp_edges_index, CDR3_gvp_edges,
                                            CDR3_gvp_batch_batch, CDR3_len_batch, 35, device)
        epitope_gvp_seq, epitope_gvp_feat = self.epitope_gvp_layer(epitope_gvp_nodes, epitope_gvp_edges_index, epitope_gvp_edges,
                                                  epitope_gvp_batch_batch, epitope_len_batch, 22, device)
        return CDR3_gvp_seq, CDR3_gvp_feat, epitope_gvp_seq, epitope_gvp_feat


class blosum62_unimodal_model(nn.Module):
    def __init__(self):
        super(blosum62_unimodal_model, self).__init__()
        self.CDR3_linear = nn.Linear(20, 64)
        self.epitope_linear = nn.Linear(20, 64)

        self.CDR3_mhsa1 = nn.MultiheadAttention(64, 4, dropout=0.1)
        self.CDR3_norm_layer1 = nn.LayerNorm(64)
        self.CDR3_ffn_mlp1 = Mlp_layer(in_features=64, hidden_features=64, act_layer=nn.GELU, drop=0.)
        self.CDR3_mhsa2 = nn.MultiheadAttention(64, 4, dropout=0.1)
        self.CDR3_norm_layer2 = nn.LayerNorm(64)
        self.CDR3_ffn_mlp2 = Mlp_layer(in_features=64, hidden_features=64, act_layer=nn.GELU, drop=0.)
        self.CDR3_norm_layer3 = nn.LayerNorm(64)

        self.epitope_mhsa1 = nn.MultiheadAttention(64, 4, dropout=0.1)
        self.epitope_norm_layer1 = nn.LayerNorm(64)
        self.epitope_ffn_mlp1 = Mlp_layer(in_features=64, hidden_features=64, act_layer=nn.GELU, drop=0.)
        self.epitope_mhsa2 = nn.MultiheadAttention(64, 4, dropout=0.1)
        self.epitope_norm_layer2 = nn.LayerNorm(64)
        self.epitope_ffn_mlp2 = Mlp_layer(in_features=64, hidden_features=64, act_layer=nn.GELU, drop=0.)
        self.epitope_norm_layer3 = nn.LayerNorm(64)

        self.drop_path = DropPath(0.1)

    def forward(self, CDR3, epitope):
        CDR3_f = self.CDR3_linear(CDR3)
        epitope_f = self.epitope_linear(epitope)

        CDR3_f = CDR3_f.permute(1, 0, 2)
        CDR3_f = CDR3_f + self.drop_path(self.CDR3_mhsa1(CDR3_f, CDR3_f, CDR3_f)[0])
        CDR3_f = CDR3_f + self.drop_path(self.CDR3_ffn_mlp1(self.CDR3_norm_layer1(CDR3_f)))
        CDR3_f = CDR3_f + self.drop_path(self.CDR3_mhsa2(CDR3_f, CDR3_f, CDR3_f)[0])
        CDR3_f = CDR3_f + self.drop_path(self.CDR3_ffn_mlp2(self.CDR3_norm_layer2(CDR3_f)))
        CDR3_seq = CDR3_f.permute(1, 0, 2)
        CDR3_feat = self.CDR3_norm_layer3(torch.sum(CDR3_seq, dim=1))

        epitope_f = epitope_f.permute(1, 0, 2)
        epitope_f = epitope_f + self.drop_path(self.epitope_mhsa1(epitope_f, epitope_f, epitope_f)[0])
        epitope_f = epitope_f + self.drop_path(self.epitope_ffn_mlp1(self.epitope_norm_layer1(epitope_f)))
        epitope_f = epitope_f + self.drop_path(self.epitope_mhsa2(epitope_f, epitope_f, epitope_f)[0])
        epitope_f = epitope_f + self.drop_path(self.epitope_ffn_mlp2(self.epitope_norm_layer2(epitope_f)))
        epitope_seq = epitope_f.permute(1, 0, 2)
        epitope_feat = self.epitope_norm_layer3(torch.sum(epitope_seq, dim=1))
        return CDR3_seq, CDR3_feat, epitope_seq, epitope_feat


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None


class fuse_feat(nn.Module):
    def __init__(self, dim=64, seq_len=22, graph_len=254):
        super(fuse_feat, self).__init__()
        hidden_dim = seq_len * dim
        ##########################################
        # mapping to same sized space
        ##########################################
        self.project_gvp = nn.Sequential(
            nn.Linear(seq_len * dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.project_graph = nn.Sequential(
            nn.Linear(graph_len * dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.project_blosum = nn.Sequential(
            nn.Linear(seq_len * dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        ##########################################
        # private encoders
        ##########################################
        self.private_gvp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.private_graph = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.private_blosum = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3)
        )

        self.sp_discriminator = nn.Sequential(nn.Linear(hidden_dim, 4))
        self.recon_gvp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim))
        self.recon_graph = nn.Sequential(nn.Linear(hidden_dim, hidden_dim))
        self.recon_blosum = nn.Sequential(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, gvp, graph, blosum):
        gvp = gvp.reshape(gvp.shape[0], -1)
        graph = graph.reshape(graph.shape[0], -1)
        blosum = blosum.reshape(blosum.shape[0], -1)

        self.shared_private(gvp, graph, blosum)

        reversed_shared_code_gvp = ReverseLayerF.apply(self.utt_shared_gvp, 1)
        reversed_shared_code_graph = ReverseLayerF.apply(self.utt_shared_graph, 1)
        reversed_shared_code_blosum = ReverseLayerF.apply(self.utt_shared_blosum, 1)

        self.domain_label_gvp = self.discriminator(reversed_shared_code_gvp)
        self.domain_label_graph = self.discriminator(reversed_shared_code_graph)
        self.domain_label_blosum = self.discriminator(reversed_shared_code_blosum)

        self.shared_or_private_p_gvp = self.sp_discriminator(self.utt_private_gvp)
        self.shared_or_private_p_graph = self.sp_discriminator(self.utt_private_graph)
        self.shared_or_private_p_blosum = self.sp_discriminator(self.utt_private_blosum)
        self.shared_or_private_s = self.sp_discriminator(
            (self.utt_shared_gvp + self.utt_shared_graph + self.utt_shared_blosum) / 3.0)

        # For reconstruction
        self.reconstruct()

        # 1-LAYER TRANSFORMER FUSION
        h = torch.stack((self.utt_private_gvp, self.utt_private_graph, self.utt_private_blosum, self.utt_shared_gvp,
                         self.utt_shared_graph, self.utt_shared_blosum), dim=0)
        h = self.transformer_encoder(h)
        h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)
        o = self.fusion(h)
        return o

    def shared_private(self, gvp, graph, blosum):
        # Projecting to same sized space
        self.utt_gvp_orig = utterance_gvp = self.project_gvp(gvp)
        self.utt_graph_orig = utterance_graph = self.project_graph(graph)
        self.utt_blosum_orig = utterance_blosum = self.project_blosum(blosum)

        # Private-shared components
        self.utt_private_gvp = self.private_gvp(utterance_gvp)
        self.utt_private_graph = self.private_graph(utterance_graph)
        self.utt_private_blosum = self.private_blosum(utterance_blosum)

        self.utt_shared_gvp = self.shared(utterance_gvp)
        self.utt_shared_graph = self.shared(utterance_graph)
        self.utt_shared_blosum = self.shared(utterance_blosum)

    def reconstruct(self):
        self.utt_gvp = (self.utt_private_gvp + self.utt_shared_gvp)
        self.utt_graph = (self.utt_private_graph + self.utt_shared_graph)
        self.utt_blosum = (self.utt_private_blosum + self.utt_shared_blosum)

        self.utt_gvp_recon = self.recon_gvp(self.utt_gvp)
        self.utt_graph_recon = self.recon_graph(self.utt_graph)
        self.utt_blosum_recon = self.recon_blosum(self.utt_blosum)


class multimodal_model(nn.Module):
    def __init__(self):
        super(multimodal_model, self).__init__()
        self.graph_model = graph_unimodal_model()
        self.gvp_model = gvp_unimodal_model()
        self.blosum62_model = blosum62_unimodal_model()
        self.out_layer = nn.Sequential(
            nn.Linear(64 * 35 + 64 * 22, int((64 * 35 + 64 * 22) / 2)),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(int((64 * 35 + 64 * 22) / 2), 2),
        )
        self.CDR3_fusion = fuse_feat(seq_len=35, graph_len=254)
        self.epitope_fusion = fuse_feat(seq_len=22, graph_len=179)

    def forward(self, CDR3_graph_batch, epitope_graph_batch,
                CDR3_blosum62_feat_batch, epitope_blosum62_feat_batch,
                CDR3_gvp_nodes, CDR3_gvp_edges, CDR3_gvp_edges_index, CDR3_gvp_batch_batch,
                epitope_gvp_nodes, epitope_gvp_edges, epitope_gvp_edges_index, epitope_gvp_batch_batch,
                CDR3_seq_len_batch, epitope_seq_len_batch,
                CDR3_molecule_len_batch, epitope_molecule_len_batch, device):
        CDR3_graph_seq, CDR3_graph_feat, epitope_graph_seq, epitope_graph_feat = self.graph_model(CDR3_graph_batch, epitope_graph_batch, CDR3_molecule_len_batch, epitope_molecule_len_batch, device)
        CDR3_gvp_seq, CDR3_gvp_feat, epitope_gvp_seq, epitope_gvp_feat = self.gvp_model(CDR3_gvp_nodes,
                                                                                        CDR3_gvp_edges,
                                                                                        CDR3_gvp_edges_index,
                                                                                        epitope_gvp_nodes,
                                                                                        epitope_gvp_edges,
                                                                                        epitope_gvp_edges_index,
                                                                                        CDR3_gvp_batch_batch,
                                                                                        epitope_gvp_batch_batch,
                                                                                        CDR3_seq_len_batch,
                                                                                        epitope_seq_len_batch, device)
        CDR3_blosum62_seq, CDR3_blosum62_feat, epitope_blosum62_seq, epitope_blosum62_feat = self.blosum62_model(
            CDR3_blosum62_feat_batch, epitope_blosum62_feat_batch)

        CDR3_f = self.CDR3_fusion(CDR3_gvp_seq, CDR3_graph_seq, CDR3_blosum62_seq)
        epitope_f = self.epitope_fusion(epitope_gvp_seq, epitope_graph_seq, epitope_blosum62_seq)
        f = torch.cat((CDR3_f, epitope_f), dim=1)
        output = self.out_layer(f)
        return output
