import numpy as np
import torch
from pytorch_lightning.metrics import Metric

from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score



def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    mapping, w = compute_best_mapping(y_true, y_pred)
    return sum([w[i, j] for i, j in mapping]) * 1.0 / y_pred.size


def compute_best_mapping(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    return np.transpose(np.asarray(linear_sum_assignment(w.max() - w))), w


class ClusterMetrics(Metric):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.add_state("preds", default=[])
        self.add_state("targets", default=[])

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self):
        preds = torch.cat(self.preds, dim=-1)
        targets = torch.cat(self.targets)
        targets -= targets.min()
        acc, nmi, ari = [], [], []
        for head in range(self.num_heads):
            t = targets.cpu().numpy()
            p = preds[head].cpu().numpy()
            acc.append(torch.tensor(cluster_acc(t, p), device=preds.device))
            nmi.append(torch.tensor(nmi_score(t, p), device=preds.device))
            ari.append(torch.tensor(ari_score(t, p), device=preds.device))
        return {"acc": acc, "nmi": nmi, "ari": ari}

# class ClusterMetrics(Metric):
#     def __init__(self, num_heads, num_classes):
#         super().__init__()
#         self.num_heads = num_heads
#         self.num_classes=num_classes
#         self.add_state("preds", default=[])
#         self.add_state("targets", default=[])
#
#     def update(self, preds: torch.Tensor, targets: torch.Tensor):
#         self.preds.append(preds)
#         self.targets.append(targets)
#
#     def compute(self):
#         preds = torch.cat(self.preds, dim=-1)
#         targets = torch.cat(self.targets)
#         tar_min = int(targets.min())
#         targets -= targets.min()
#         acc, nmi, ari = [], [], []
#         # acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7 = [], [], [], [], [], [], []
#         for head in range(self.num_heads):
#             t = targets.cpu().numpy()
#             p = preds[head].cpu().numpy()
#             acc.append(torch.tensor(cluster_acc(t, p), device=preds.device))
#             nmi.append(torch.tensor(nmi_score(t, p), device=preds.device))
#             ari.append(torch.tensor(ari_score(t, p), device=preds.device))
#
#             # acc for each class
#             acc_class = [0.0] * self.num_classes
#             for head in range(self.num_heads):
#                 t = targets.cpu().numpy()
#                 p = preds[head].cpu().numpy()
#                 mapping, w = compute_best_mapping(t, p)
#                 w_sum = w.sum(axis=0)
#                 for i, j in mapping:
#                     if j + tar_min < self.num_classes:
#                         acc_class[j + tar_min] += w[i, j] * 1.0 / w_sum[j]
#             for i in range(7):
#                 acc_class[i] = acc_class[i] / self.num_heads
#
#         return {"acc": acc, "nmi": nmi, "ari": ari, "acc_each_class": acc_class}
