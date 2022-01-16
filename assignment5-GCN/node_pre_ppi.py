import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.optim import lr_scheduler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, PairNorm
from torch_geometric.utils.undirected import to_undirected
import random
import matplotlib.pyplot as plt

import os.path as osp

from sklearn.metrics import accuracy_score
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T


def data_loader():
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'dataset', 'ppi2')
    # pre_transform = T.Compose([T.GCNNorm(), T.ToSparseTensor()])
    train_dataset = PPI(path, split='train')
    val_dataset = PPI(path, split='val')
    test_dataset = PPI(path, split='test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader


def drop_edge(edge_index, keep_ratio: float = 1.):
    num_keep = int(keep_ratio * edge_index.shape[1])
    temp = [True] * num_keep + [False] * (edge_index.shape[1] - num_keep)
    random.shuffle(temp)
    return edge_index[:, temp]


class GCNNodeClassifier(torch.nn.Module):
    def __init__(self,
                 dim_features,
                 num_classes,
                 num_layers,
                 add_self_loops: bool = True,
                 use_pairnorm: bool = False,
                 drop_edge: float = 1.,
                 activation: str = 'relu',
                 # undirected: bool = False
                 ):
        super(GCNNodeClassifier, self).__init__()
        dim_hidden = 2048

        self.gconvs = torch.nn.ModuleList(
            [GCNConv(in_channels=dim_features, out_channels=dim_hidden, add_self_loops=add_self_loops)]
            + [GCNConv(in_channels=dim_hidden, out_channels=dim_hidden, add_self_loops=add_self_loops) for i in
               range(num_layers - 2)]
        )
        self.final_conv = GCNConv(in_channels=dim_hidden, out_channels=num_classes, add_self_loops=add_self_loops)

        self.use_pairnorm = use_pairnorm
        if self.use_pairnorm:
            self.pairnorm = PairNorm()
        self.drop_edge = drop_edge
        activations_map = {'relu': torch.relu, 'tanh': torch.tanh, 'sigmoid': torch.sigmoid,
                           'leaky_relu': torch.nn.LeakyReLU(0.1)}
        self.activation_fn = activations_map[activation]

    def forward(self, x, edge_index):
        for l in self.gconvs:
            edges = drop_edge(edge_index, self.drop_edge).to(device)
            x = l(x, edges)
            if self.use_pairnorm:
                x = self.pairnorm(x)
            x = self.activation_fn(x)
        x = self.final_conv(x, edge_index)

        return x


def eval_acc(y_pred, y):
    return ((torch.argmax(y_pred, dim=-1) == y).float().sum() / y.shape[0]).item()


num_epochs = 1000
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

train_loader, val_loader, test_loader = data_loader()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = torch.nn.BCEWithLogitsLoss()
# 多标签分类任务多用

def train(scheduler):
    model.train()

    total_loss, total_examples = 0, 0
    ys, preds = [], []
    for data in train_loader:
        # data = get_data(data[0])
        ys.append(data.y)
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        preds.append((out > 0).float().cpu())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes

    scheduler.step()
    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return total_loss / total_examples, accuracy_score(y, pred)

def test():
    model.eval()

    ys, preds = [], []
    for loader in [val_loader, test_loader]:
        for data in loader:
            ys.append(data.y)
            out = model(data.x.to(device), data.edge_index.to(device))
            preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return accuracy_score(y, pred)


for i_case, kwargs in enumerate(test_cases):
    print(f'Test Case {i_case:>2}')
    model = GCNNodeClassifier(50, 121, **kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.6)

    history_test_acc = []

    for i_epoch in range(0, num_epochs):

        loss, train_acc = train(exp_lr_scheduler)
        test_acc = test()

        history_test_acc.append(test_acc)
        # if (i_epoch+1) % 10 == 0:
        print(f'Epoch {(i_epoch+1):04d} Loss = {loss:.4f}. Train Acc = {train_acc:.4f}. Test Acc = {test_acc:.4f}')

    kwargs['best_acc'] = max(history_test_acc)
    plt.plot(list(range(num_epochs)), history_test_acc, label=f'case_{str(i_case).zfill(2)}')

plt.legend()
plt.savefig('ppi-HistoryAcc.jpg')
pd.DataFrame(test_cases).to_csv('ppi-Result.csv')

