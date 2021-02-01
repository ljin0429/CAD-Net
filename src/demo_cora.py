import argparse
import torch
import torch.nn.functional as F
import time
from torch_geometric.utils import add_self_loops
from AdaCAD_cora import AdaCAD
from datasets import get_planetoid_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--is_debug', type=bool, default=False)
parser.add_argument('--runs', type=int, default=100)
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_drop', type=bool, default=True)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--add_selfloops', type=bool, default=True)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--K', type=int, default=6)
parser.add_argument('--beta', type=float, default=0.8)
parser.add_argument('--entropy_regularization', type=float, default=0.5)
args = parser.parse_args()


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.embed1 = torch.nn.Linear(dataset.num_features, args.hidden)
        self.embed2 = torch.nn.Linear(args.hidden, dataset.num_classes)
        self.adgs = AdaCAD(K=args.K,
                           beta=args.beta,
                           dropout=args.dropout
                           )

    def reset_parameters(self):
        self.embed1.reset_parameters()
        self.embed2.reset_parameters()

    def forward(self, data, is_debug):

        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=args.dropout, training=self.training)

        train_mask = data.train_mask

        # Add self-loops to the adjacency matrix.
        if args.add_selfloops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # ========================= #
        # === Feature Transform === #
        # ========================= #
        # 1st layer
        x = self.embed1(x)
        x = F.leaky_relu(x, 0.05)

        # 2nd layer
        x = self.embed2(x)

        # ================================ #
        # ============ AggCAD ============ #
        # ================================ #
        x, ent, debug_tensor = self.adgs(x, edge_index, train_mask, is_debug)

        return F.log_softmax(x, dim=1), ent, debug_tensor

    def __repr__(self):
        return self.__class__.__name__


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out, ent, _ = model(data, is_debug=False)

    class_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    ent_loss = ent.mean()

    loss = class_loss + (args.entropy_regularization * ent_loss)

    loss.backward()
    optimizer.step()


def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        logits, _, _ = model(data, is_debug=False)

    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]

        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs['{}_loss'.format(key)] = loss
        outs['{}_acc'.format(key)] = acc

    return outs


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
    data = dataset[0]
    model = Net(dataset)

    val_losses, accs, durations = [], [], []
    for run_num in range(args.runs):
        data = data.to(device)
        model.to(device).reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay,
                                     amsgrad=True)

        if args.lr_drop:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                        step_size=50,
                                                        gamma=0.5)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        best_val_loss = float('inf')
        test_acc = 0
        val_loss_history = []

        for epoch in range(1, args.epochs + 1):
            train(model, optimizer, data)
            eval_info = evaluate(model, data)
            eval_info['epoch'] = epoch

            if eval_info['val_loss'] < best_val_loss:
                best_val_loss = eval_info['val_loss']
                test_acc = eval_info['test_acc']

            val_loss_history.append(eval_info['val_loss'])
            if args.early_stopping > 0 and epoch > args.epochs // 2:
                tmp = torch.tensor(val_loss_history[-(args.early_stopping + 1):-1])
                if eval_info['val_loss'] > tmp.mean().item():
                    break

            if args.lr_drop:
                scheduler.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()

        # Average training time per epoch (ms)
        duration_per_epoch = (t_end - t_start) / epoch * 1000

        val_losses.append(best_val_loss)
        accs.append(test_acc)
        durations.append(duration_per_epoch)

        print('Run: {:d}, Val Loss: {:.3f}, Test Accuracy: {:.3f}, Time(ms): {:.2f}'.
              format(run_num + 1,
                     best_val_loss,
                     test_acc,
                     duration_per_epoch))

    loss, acc, duration = torch.tensor(val_losses), torch.tensor(accs), torch.tensor(durations)

    print('============================= Total =============================')
    print('Val Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Time(ms): {:.2f}'.
          format(loss.mean().item(),
                 acc.mean().item(),
                 acc.std().item(),
                 duration.mean().item()))


