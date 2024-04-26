import torch
import torch.nn.functional as F
from torch import nn

from model.GCRNCell import GCRNCell
from model.Layer import RelatinoShipConv, SignalDecomposition


class GCRN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, embed_dim, cheb_k, num_layers):
        super(GCRN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(GCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(GCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x, init_state, node_embeddings):
        # x: (B, T, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []

        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)

        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)


class STBlock(nn.Module):
    def __init__(self, args):
        super(STBlock, self).__init__()
        self.hidden_dim = args.rnn_units
        self.num_nodes = args.num_nodes
        self.encoder = GCRN(args.num_nodes, args.input_dim, args.rnn_units, args.embed_dim, args.cheb_k, args.num_layers)
        self.RSConv = RelatinoShipConv(args.input_dim, args.rnn_units)  # 更改
        self.layers = args.num_layers

        if self.layers == 1:
            self.end_conv1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=args.horizon, kernel_size=(1, 1), bias=True),
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            )
        else:
            self.end_conv1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.end_conv2 = nn.Conv2d(in_channels=args.horizon, out_channels=args.horizon, kernel_size=(1, args.rnn_units), bias=True)
        self.end_conv3 = nn.Conv2d(in_channels=args.horizon, out_channels=args.horizon, kernel_size=(1, 2), dilation=(1, self.hidden_dim), bias=True)

        self.dropout = nn.Dropout(args.dropout)
        self.conv = nn.Conv2d(in_channels=args.horizon, out_channels=args.horizon, kernel_size=(1, self.hidden_dim + 1), bias=False)

    def forward(self, x, node_embeddings):
        batch = x.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(self.num_nodes).to(supports.device), supports]
        supports = torch.stack(support_set, dim=0)
        init_state = self.encoder.init_hidden(x.shape[0])
        output, _ = self.encoder(x, init_state, node_embeddings)
        if self.layers == 1:  # 因为层数为1, 预测的第一个时间步是有问题的
            output = self.dropout(output[:, -1:, :, :])
        else:
            output = self.dropout(output)
        output = self.end_conv1(output)

        relation = self.RSConv(x)
        relation_GCN = torch.einsum('btnc,knn->kbcnt', relation, supports)
        relation_GCN = torch.permute(relation_GCN, (1, 4, 3, 2, 0))
        relation_GCN = torch.reshape(relation_GCN, (batch, 12, self.num_nodes, -1))
        relation_ST = self.conv(relation_GCN)

        output = self.end_conv2(torch.cat((relation_ST, output), dim=-1))
        output = self.end_conv3(output)

        return output

class SDSTGCN(nn.Module):
    def __init__(self, args):
        super(SDSTGCN, self).__init__()
        self.inh_model = STBlock(args)
        self.dif_model = STBlock(args)
        self.node_embeddings_inh = nn.Parameter(torch.randn(args.num_nodes, args.embed_dim), requires_grad=True)  # 固定信号
        self.node_embeddings_dif = nn.Parameter(torch.randn(args.num_nodes, args.embed_dim), requires_grad=True)  # 扩散信号
        self.T_i_D_emb = nn.Parameter(torch.empty(288, args.embed_dim))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, args.embed_dim))
        self.SignalDecomposition = SignalDecomposition(args.embed_dim, args.embed_dim, args.embed_dim, args.rnn_units)

    def forward(self, source):
        # source (B, T, N, D)

        t_i_d_data = source[..., 1]
        T_i_D_emb = self.T_i_D_emb[(t_i_d_data * 288).type(torch.LongTensor)]
        d_i_w_data = source[..., 2]
        D_i_W_emb = self.D_i_W_emb[(d_i_w_data).type(torch.LongTensor)]

        inh = self.SignalDecomposition(self.node_embeddings_inh, self.node_embeddings_dif, source[..., :1], T_i_D_emb, D_i_W_emb)
        inh_data = torch.zeros_like(source)
        inh_data[..., :1] = inh
        inh_data[..., 1:] = source[..., 1:]
        inh_output = self.inh_model(inh_data, self.node_embeddings_inh)

        dif_data = torch.zeros_like(source)
        dif_data[..., :1] = inh_data[..., :1]
        dif_data = source - dif_data
        dif_output = self.dif_model(dif_data, self.node_embeddings_dif)
        return dif_output + inh_output