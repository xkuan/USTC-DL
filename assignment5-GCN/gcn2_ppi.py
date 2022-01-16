import os.path as osp
from datetime import datetime

import torch
from torch.nn import Linear
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch_geometric.datasets import PPI
import torch_geometric.transforms as T
from torch_geometric.nn import GCN2Conv
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'dataset', 'ppi')
pre_transform = T.Compose([T.GCNNorm(), T.ToSparseTensor()])
train_dataset = PPI(path, split='train', pre_transform=pre_transform)
val_dataset = PPI(path, split='val', pre_transform=pre_transform)
test_dataset = PPI(path, split='test', pre_transform=pre_transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


class Net(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(train_dataset.num_features, hidden_channels))
        self.lins.append(Linear(hidden_channels, train_dataset.num_classes))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

    def forward(self, x, adj_t):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            h = F.dropout(x, self.dropout, training=self.training)
            h = conv(h, x_0, adj_t)
            x = h + x
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(hidden_channels=2048, num_layers=9, alpha=0.5, theta=1.0,
            shared_weights=False, dropout=0.2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()

def train():
    model.train()

    total_loss, total_examples = 0, 0
    ys, preds = [], []
    for data in train_loader:
        ys.append(data.y)
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.adj_t)
        loss = criterion(out, data.y)
        preds.append((out > 0).float().cpu())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return total_loss / total_examples, accuracy_score(y, pred)


@torch.no_grad()
def test(loader):
    model.eval()

    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        out = model(data.x.to(device), data.adj_t.to(device))
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return accuracy_score(y, pred)


writer = SummaryWriter(f"gcn2_logs/{int(datetime.timestamp(datetime.now()))}")
for epoch in range(1, 2001):
    loss, train_acc = train()
    val_acc = test(val_loader)
    writer.add_scalar('loss', loss, epoch)
    writer.add_scalar('accuracy', val_acc, epoch)
    print(f'Epoch: {epoch:04d}, Loss: {loss:.4f}, train_acc:{train_acc:.4f}, val_acc: {val_acc:.4f}')

writer.close()
test_acc = test(test_loader)
print(f'Finish! test_acc: {test_acc:.4f}')

# tensorboard --logdir=link_pre_logs