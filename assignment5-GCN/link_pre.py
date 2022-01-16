# https://github.com/Orbifold/pyg-link-prediction

import random

import numpy as np
from datetime import datetime

import pandas as pd
import torch
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, PairNorm
from sklearn.metrics import accuracy_score
from torch_geometric.loader import NeighborLoader
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from node_pre import get_data
from torch_geometric.datasets import PPI


def data_loader(data_name, batch_size=16):
    if data_name in ['cora', 'citeseer']:
        data, _ = get_data(data_name)
        train_loader = NeighborLoader(data, num_neighbors=[10]*2, shuffle=True, input_nodes=data.train_mask, batch_size=batch_size)
        test_loader = NeighborLoader(data, num_neighbors=[10]*2, input_nodes=data.test_mask, batch_size=batch_size)

    elif data_name == 'ppi':
        path = './dataset/ppi2'
        train_dataset = PPI(path, split='train')
        test_dataset = PPI(path, split='test')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return data, train_loader, test_loader


def drop_edge(edge_index, keep_ratio: float = 1.):
    num_keep = int(keep_ratio * edge_index.shape[1])
    temp = [True] * num_keep + [False] * (edge_index.shape[1] - num_keep)
    random.shuffle(temp)
    return edge_index[:, temp]

# the actual Pyg network
class GCNLinkPrediction(torch.nn.Module):
    def __init__(self,
                 dim_features,
                 num_classes,
                 num_layers,
                 add_self_loops: bool = True,
                 use_pairnorm: bool = False,
                 drop_edge: float = 1.,
                 activation: str = 'relu',
                 ):
        super(GCNLinkPrediction, self).__init__()
        dim_hidden = 128

        self.gconvs = torch.nn.ModuleList(
            [GCNConv(in_channels=dim_features, out_channels=dim_hidden, add_self_loops=add_self_loops)]
            + [GCNConv(in_channels=dim_hidden, out_channels=dim_hidden, add_self_loops=add_self_loops)
               for i in range(num_layers - 2)]
        )
        self.final_conv = GCNConv(
            in_channels=dim_hidden, out_channels=num_classes, add_self_loops=add_self_loops)

        self.use_pairnorm = use_pairnorm
        if self.use_pairnorm:
            self.pairnorm = PairNorm()
        self.drop_edge = drop_edge
        activations_map = {'relu': torch.relu, 'tanh': torch.tanh, 'sigmoid': torch.sigmoid,
                           'leaky_relu': torch.nn.LeakyReLU(0.1)}
        self.activation_fn = activations_map[activation]


    def encode(self, x, edge_index):
        for l in self.gconvs:
            edges = drop_edge(edge_index, self.drop_edge)
            x = l(x, edges)
            if self.use_pairnorm:
                x = self.pairnorm(x)
            x = self.activation_fn(x)
        x = self.final_conv(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        # cosine similarity
        edge_label_index = edge_label_index.type(torch.long)
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim = -1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple = False).t()


def train(train_loader, model, optimizer):

    criterion = torch.nn.BCEWithLogitsLoss()
    model.train()
    total_examples = total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        batch_size = batch.batch_size
        z = model.encode(batch.x, batch.edge_index)
        neg_edge_index = negative_sampling(
            edge_index = batch.edge_index, num_nodes = batch.num_nodes, num_neg_samples = None, method = 'sparse')
        neg_edge_index = neg_edge_index.to(device)
        edge_label_index = torch.cat([batch.edge_index, neg_edge_index], dim = -1, )
        edge_label = torch.cat([torch.ones(batch.edge_index.size(1)), torch.zeros(neg_edge_index.size(1))], dim = 0)
        edge_label = edge_label.to(device)
        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()
        total_examples += batch_size
        total_loss += float(loss) * batch_size
    return total_loss / total_examples


@torch.no_grad()
def test(loader, model):

    model.eval()
    scores = []
    threshold = torch.tensor([0.7])
    for batch in loader:
        if batch.edge_index.size(1) == 0:
            break
        batch = batch.to(device)
        z = model.encode(batch.x, batch.edge_index)
        out = model.decode(z, batch.edge_index).view(-1).sigmoid().cpu()
        pred = (out > threshold).float() * 1
        score = accuracy_score(np.ones(batch.edge_index.size(1)), pred.cpu().numpy())
        scores.append(score)
    return np.average(scores)


test_cases = [
    {'num_layers':2, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':1., 'activation':'relu'},
    # num layers
    {'num_layers':4, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':1., 'activation':'relu'},
    {'num_layers':6, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':1., 'activation':'relu'},
    # self loop
    {'num_layers':2, 'add_self_loops':False, 'use_pairnorm':False, 'drop_edge':1., 'activation':'relu'},
    # pair norm
    {'num_layers':2, 'add_self_loops':True, 'use_pairnorm':True, 'drop_edge':1., 'activation':'relu'},
    {'num_layers':4, 'add_self_loops':True, 'use_pairnorm':True, 'drop_edge':1., 'activation':'relu'},
    {'num_layers':6, 'add_self_loops':True, 'use_pairnorm':True, 'drop_edge':1., 'activation':'relu'},
    # drop edge
    {'num_layers':2, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':0.6, 'activation':'relu'},
    {'num_layers':4, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':0.6, 'activation':'relu'},
    # activation fn
    {'num_layers':2, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':1., 'activation':'tanh'},
    {'num_layers':2, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':1., 'activation':'leaky_relu'},
]

def run(data_name, epochs=100):

    run_id = int(datetime.timestamp(datetime.now()))
    writer = SummaryWriter(f"link_pre_logs/{data_name}-{run_id}")

    batch_size = 16
    data, train_loader, test_loader = data_loader(data_name, batch_size)

    best_acc = []
    for i_case, kwargs in enumerate(test_cases):
        print(f'Test Case {i_case:>2}')
        model = GCNLinkPrediction(data.num_features, 64, **kwargs).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
        history_test_acc = []

        for epoch in range(0, epochs):
            for i in range(1):
                loss = train(train_loader, model, optimizer)
                val_acc = test(test_loader, model)
                history_test_acc.append(val_acc)
                writer.add_scalar('train_loss', loss, epoch)
                writer.add_scalar('val_accuracy', val_acc, epoch)
                print(f"Epoch: {epoch+1:03d}, Train Loss: {loss:.4f}, Val Acc: {val_acc:.4f}")

        best_acc.append(max(history_test_acc))
    return best_acc
    writer.close()



cora_best_acc = run('cora', epochs=100)
citeseer_best_acc = run('citeseer', epochs=100)
result = pd.DataFrame(test_cases)
result['cora_best_acc'] = cora_best_acc
result['citeseer_best_acc'] = citeseer_best_acc
# result.to_csv('LP_result.csv')
# tensorboard --logdir=link_pre_logs --port=6007