import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.metrics import Accuracy
import numpy as np
from torchvision import models
import torch.nn as nn

import os
os.environ["WANDB_API_KEY"] = 'a7ccbe87671159f0147c1baf72b1da302ecec4dd'


from utils.data import get_datamodule
from utils.nets import MMCXRNCD
from utils.eval import ClusterMetrics, cluster_acc
from utils.pseudo_label_generation import SinkhornKnopp, js_divergence


import numpy as np
from argparse import ArgumentParser
from datetime import datetime

def kl_divergence(matrix1, matrix2):
    eps = 1e-6  # 避免除以0
    matrix1[matrix1 == 0] = eps
    matrix2[matrix2 == 0] = eps
    divergence = np.sum(matrix1 * np.log(matrix1 / matrix2))
    return divergence



parser = ArgumentParser()
parser.add_argument("--dataset", default="MIMIC", type=str, help="dataset")
parser.add_argument("--download", default=False, action="store_true", help="wether to download")
parser.add_argument("--log_dir", default="./logs", type=str, help="log directory")
parser.add_argument("--batch_size", default=128, type=int, help="batch size")
parser.add_argument("--num_workers", default=10, type=int, help="number of workers")
parser.add_argument("--arch", default="resnet50", type=str, help="backbone architecture")
parser.add_argument("--base_lr", default=1e-4, type=float, help="learning rate")
parser.add_argument("--min_lr", default=1e-6, type=float, help="min learning rate")
parser.add_argument("--momentum_opt", default=0.9, type=float, help="momentum for optimizer")
parser.add_argument("--weight_decay_opt", default=1.5e-6, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="warmup epochs")
parser.add_argument("--proj_dim", default=256, type=int, help="projected dim")
parser.add_argument("--hidden_dim", default=2048, type=int, help="hidden dim in proj/pred head")
parser.add_argument("--overcluster_factor", default=3, type=int, help="overclustering factor")
parser.add_argument("--num_heads", default=4, type=int, help="number of heads for clustering")
parser.add_argument("--num_hidden_layers", default=1, type=int, help="number of hidden layers")
parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")
parser.add_argument("--epsilon_sk", default=0.05, type=float, help="epsilon for the Sinkhorn")
parser.add_argument("--temperature", default=0.1, type=float, help="softmax temperature")
parser.add_argument("--comment", default=datetime.now().strftime("%b%d_%H-%M-%S"), type=str)
parser.add_argument("--project", default="MMNCD", type=str, help="wandb project")
parser.add_argument("--entity", default=None, type=str, help="wandb entity")
parser.add_argument("--offline", default=False, action="store_true", help="disable wandb")
parser.add_argument("--num_labeled_classes", default=4, type=int, help="number of labeled classes")
parser.add_argument("--num_unlabeled_classes", default=4, type=int, help="number of unlab classes")
parser.add_argument("--pretrained", default="cxr",type=str, help="pretrained checkpoint path")
parser.add_argument("--multicrop", default=False, action="store_true", help="activates multicrop")
parser.add_argument("--num_large_crops", default=2, type=int, help="number of large crops")
parser.add_argument("--num_small_crops", default=2, type=int, help="number of small crops")



class Discoverer(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters({k: v for (k, v) in kwargs.items() if not callable(v)})

        # build model,
        self.model = MMCXRNCD(
            arch=self.hparams.arch,
            low_res=False,
            num_labeled=self.hparams.num_labeled_classes,
            num_unlabeled=self.hparams.num_unlabeled_classes,
            proj_dim=self.hparams.proj_dim,
            hidden_dim=self.hparams.hidden_dim,
            overcluster_factor=self.hparams.overcluster_factor,
            num_heads=self.hparams.num_heads,
            num_hidden_layers=self.hparams.num_hidden_layers,
        )

        #load params from pretrained gloria
        image_pretrained = torch.load('./pretrained/sub1_set1_resnet.pth')
        text_pretrained = torch.load('./pretrained/sub1_set1_bio.pth')

        self.model.image_branch.img_encoder.load_state_dict(image_pretrained.state_dict(), strict=False)
        self.model.text_branch.text_encoder.load_state_dict(text_pretrained.state_dict(), strict=False)

        # freeze text encoder
        # for param in self.model.text_branch.text_encoder.parameters():
        #     param.requires_grad = False

        # Sinkorn-Knopp
        self.sk = SinkhornKnopp(
            num_iters=self.hparams.num_iters_sk, epsilon=self.hparams.epsilon_sk
        )   #pseudo label assignment1


        # metrics
        self.metrics = torch.nn.ModuleList(
            [
                ClusterMetrics(self.hparams.num_heads),
                ClusterMetrics(self.hparams.num_heads),
                Accuracy(),
            ]
        )
        self.metrics_inc = torch.nn.ModuleList(
            [
                ClusterMetrics(self.hparams.num_heads),
                ClusterMetrics(self.hparams.num_heads),
                Accuracy(),
            ]
        )

        # buffer for best head tracking
        self.register_buffer("img_loss_per_head", torch.zeros(self.hparams.num_heads))
        self.register_buffer("text_loss_per_head", torch.zeros(self.hparams.num_heads))



    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.base_lr,
            momentum=self.hparams.momentum_opt,
            weight_decay=self.hparams.weight_decay_opt,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
            warmup_start_lr=self.hparams.min_lr,
            eta_min=self.hparams.min_lr,
        )
        return [optimizer], [scheduler]

    def cross_entropy_loss(self, preds, targets):
        preds = F.log_softmax(preds / self.hparams.temperature, dim=-1)
        return torch.mean(-torch.sum(targets * preds, dim=-1), dim=-1)


    #x: image, y: text
    def forward(self, x, y):
        return self.model(x, y)

    def on_epoch_start(self):
        self.img_loss_per_head = torch.zeros_like(self.img_loss_per_head)
        self.text_loss_per_head = torch.zeros_like(self.text_loss_per_head)

    def unpack_batch(self, batch):
        #no multiview
        images, texts, labels = batch
        mask_lab = labels < self.hparams.num_labeled_classes
        return images, texts, labels, mask_lab



    def training_step(self, batch, _):
        images, texts, labels, mask_lab = self.unpack_batch(batch)
        nlc = self.hparams.num_labeled_classes

        # normalize prototypes
        self.model.image_branch.normalize_prototypes()
        self.model.text_branch.normalize_prototypes()

        # forward
        img_outputs, text_outputs = self.model(images, texts)

        lab_img_logits = img_outputs['img_logits_lab'][mask_lab, :]
        lab_text_logits = text_outputs['text_logits_lab'][mask_lab, :]

        # gather outputs
        img_outputs['img_logits_lab'] = (
            img_outputs['img_logits_lab'].unsqueeze(0).expand(self.hparams.num_heads, -1, -1)
        )
        text_outputs['text_logits_lab'] = (
            text_outputs['text_logits_lab'].unsqueeze(0).expand(self.hparams.num_heads,-1, -1)
        )

        img_logits = torch.cat([img_outputs['img_logits_lab'], img_outputs['img_logits_unlab']], dim=-1)
        img_logits_over = torch.cat([img_outputs['img_logits_lab'], img_outputs['img_logits_unlab_over']], dim=-1)

        text_logits = torch.cat([text_outputs['text_logits_lab'], text_outputs['text_logits_unlab']], dim=-1)
        text_logits_over = torch.cat([text_outputs['text_logits_lab'], text_outputs['text_logits_unlab_over']], dim=-1)


        img_feats = img_outputs['img_embs'] #.unsqueeze(0).repeat(self.hparams.num_heads,1,1)
        text_feats = text_outputs['text_embs'] #.unsqueeze(0).repeat(self.hparams.num_heads,1,1)

        lab_img_feats = img_feats[mask_lab,:]
        lab_text_feats = text_feats[mask_lab,:]
        lab_img_feats_sim = torch.matmul(lab_img_feats, lab_img_feats.transpose(-1, -2))  #a,a
        lab_text_feats_sim = torch.matmul(lab_text_feats, lab_text_feats.transpose(-1,-2))


        lab_img_logits_sim = torch.matmul(lab_img_logits, lab_img_logits.transpose(-1, -2))  #a,a
        lab_text_logits_sim = torch.matmul(lab_text_logits, lab_text_logits.transpose(-1,-2))

        lab_img_consistency = 1/(js_divergence(lab_img_feats_sim, lab_img_logits_sim)+0.001)

        lab_text_consistency = 1/(js_divergence(lab_text_feats_sim, lab_text_logits_sim)+0.001)




        # create targets
        targets_lab = (
            F.one_hot(labels[mask_lab], num_classes=self.hparams.num_labeled_classes)
            .float()
            .to(self.device)
        )
        targets = torch.zeros_like(img_logits)
        targets_over = torch.zeros_like(img_logits_over)


        targets_text = torch.zeros_like(img_logits)
        targets_over_text = torch.zeros_like(text_logits_over)

        targets_img = torch.zeros_like(img_logits)
        targets_over_img = torch.zeros_like(img_logits_over)


        # generate pseudo-labels with sinkhorn-knopp and fill unlab targets
        for h in range(self.hparams.num_heads):
            targets_img[h, mask_lab, :nlc] = targets_lab.type_as(targets)
            targets_over_img[h, mask_lab, :nlc] = targets_lab.type_as(targets)
            # distinguish labeled and unlabeled samples by label
            targets_img[h, ~mask_lab, nlc:] = self.sk(
                logits=img_outputs["img_logits_unlab"][h, ~mask_lab]
            ).type_as(targets)
            targets_over_img[h, ~mask_lab, nlc:] = self.sk(
                logits=img_outputs["img_logits_unlab_over"][h, ~mask_lab]
            ).type_as(targets)

            targets_text[h, mask_lab, :nlc] = targets_lab.type_as(targets)
            targets_over_text[h, mask_lab, :nlc] = targets_lab.type_as(targets)
            # distinguish labeled and unlabeled samples by label
            targets_text[h, ~mask_lab, nlc:] = self.sk(
                logits=text_outputs["text_logits_unlab"][h, ~mask_lab]
            ).type_as(targets)
            targets_over_text[h, ~mask_lab, nlc:] = self.sk(
                logits=text_outputs['text_logits_unlab_over'][h, ~mask_lab]
            ).type_as(targets)

        # emb structure similarity
        img_embs = img_feats[~mask_lab, :]
        text_embs = text_feats[~mask_lab, :]
        img_embs_sim = torch.matmul(img_embs, img_embs.transpose(-1, -2))  # a, a, a: number of unlabeled samples in a batch
        text_embs_sim = torch.matmul(text_embs, text_embs.transpose(-1, -2))  # a, a


        #pseudo label structure similarity
        img_pls = targets_img[:, ~mask_lab, nlc:]
        text_pls = targets_text[:,~mask_lab, nlc:]
        img_pls_sim = torch.matmul(img_pls, img_pls.transpose(-1, -2))  # num_heads, a, a
        text_pls_sim = torch.matmul(text_pls, text_pls.transpose(-1, -2))  # num_heads, a, a


        # normalize, not necessary
        img_embs_sim = F.normalize(img_embs_sim, dim=-1)
        text_embs_sim = F.normalize(text_embs_sim, dim=-1)

        img_pls_sim = F.normalize(img_pls_sim, dim=-1)
        text_pls_sim = F.normalize(text_pls_sim, dim=-1)


        unlab_batchsize = img_pls_sim.shape[1]
        img_consist = torch.zeros(self.hparams.num_heads, unlab_batchsize)
        text_consist = torch.zeros(self.hparams.num_heads, unlab_batchsize)
        import math

        for i in range(self.hparams.num_heads):
            for j in range(unlab_batchsize):
                # print(10*js_divergence(img_embs_sim[j].unsqueeze(0), img_pls_sim[i][j].unsqueeze(0)))
                # print(10*js_divergence(text_embs_sim[j].unsqueeze(0), text_pls_sim[i][j].unsqueeze(0)))
                img_consist[i][j] = max(0.1, 1 - 1000*js_divergence(img_embs_sim[j].unsqueeze(0), img_pls_sim[i][j].unsqueeze(0)))
                # img_consist[i][j] = min(1, img_consist[i][j])
                text_consist[i][j] = max(0.1, 1 - 1000*js_divergence(text_embs_sim[j].unsqueeze(0), text_pls_sim[i][j].unsqueeze(0)))
                # text_consist[i][j] = min(1, text_consist[i][j])
        print(img_consist[1][1])


        #convert high-confidence pl to one-hot
        threshold = 0.75
        unlabs = targets_text[h, :, nlc:].shape[1]
        for h in range(self.hparams.num_heads):
            for j in range(unlabs):
                if (torch.max(targets_img[h,j, nlc:])>threshold) and (torch.max(targets_text[h,j, nlc:])>threshold):
                    if targets_img[h,j, nlc:].argmax(dim=-1) == targets_text[h,j, nlc:].argmax(dim=-1):
                        index=targets_img[h,j, nlc:].argmax(dim=-1).item()
                        targets_img[h,j,nlc+index] = 1.0
                        targets_text[h,j,nlc+index] = 1.0
                        for i in range(self.hparams.num_unlabeled_classes):
                            if targets_img[h,j,nlc+i] < threshold:
                                targets_img[h,j,nlc+i] = 0.0
                                targets_text[h,j,nlc+i] = 0.0


        #weightning pseudo labels
        targets_img_unlab = targets_img[:, ~mask_lab, nlc:]
        targets_img_over_unlab = targets_over_img[:, ~mask_lab, nlc:]
        targets_text_unlab = targets_text[:, ~mask_lab, nlc:]
        targets_text_over_unlab = targets_over_text[:, ~mask_lab, nlc:]

        wei_imgs = torch.zeros(self.hparams.num_heads, unlab_batchsize).to(self.device)
        wei_texts = torch.zeros(self.hparams.num_heads, unlab_batchsize).to(self.device)
        # wei_imgs_over = torch.zeros(self.hparams.num_heads, unlab_batchsize).to(self.device)
        # wei_texts_over = torch.zeros(self.hparams.num_heads, unlab_batchsize).to(self.device) #(4,41)

        for h in range(self.hparams.num_heads):
            for j in range(unlab_batchsize):
                wei_imgs[h][j] = img_consist[h][j]/(img_consist[h][j]+text_consist[h][j])
                wei_texts[h][j] = text_consist[h][j] / (img_consist[h][j] + text_consist[h][j])
                # wei_imgs_over[h][j] = text_over_consist[h][j] / (img_over_consist[h][j] + text_over_consist[h][j])
                # wei_texts_over[h][j] = img_over_consist[h][j] / (img_over_consist[h][j] + text_over_consist[h][j])
            targets[h, mask_lab, :nlc] = targets_lab.type_as(targets)
            targets_over[h, mask_lab, :nlc] = targets_lab.type_as(targets_over)
        targets[:, ~mask_lab, nlc:] = (wei_imgs.unsqueeze(-1)*targets_img_unlab+wei_texts.unsqueeze(-1)*targets_text_unlab).type_as(targets)
        # targets_over[:, ~mask_lab, nlc:] = (wei_imgs_over.unsqueeze(-1)*targets_img_over_unlab+wei_texts_over.unsqueeze(-1)*targets_text_over_unlab).type_as(targets_over)
        targets_over[:, ~mask_lab, nlc:] = 0.5*(targets_img_over_unlab+targets_text_over_unlab)

        img_loss_cluster = self.cross_entropy_loss(img_logits, targets)
        img_loss_overcluster = self.cross_entropy_loss(img_logits_over, targets_over)
        text_loss_cluster = self.cross_entropy_loss(text_logits, targets)
        text_loss_overcluster = self.cross_entropy_loss(text_logits_over, targets_over)

        # update best head tracker
        self.img_loss_per_head += img_loss_cluster.clone().detach()
        self.text_loss_per_head += text_loss_cluster.clone().detach()

        # total loss
        img_loss_cluster = img_loss_cluster.mean()
        img_loss_overcluster = img_loss_overcluster.mean()
        if torch.isnan(img_loss_overcluster).any():
            print(True)
            img_loss = img_loss_cluster
        else:
            img_loss = (img_loss_cluster+img_loss_overcluster)/2
        text_loss_cluster = text_loss_cluster.mean()
        text_loss_overcluster = text_loss_overcluster.mean()
        if torch.isnan(text_loss_overcluster).any():
            print(True)
            text_loss = text_loss_cluster
        else:
            text_loss = (text_loss_cluster+text_loss_overcluster)/2
        loss = img_loss+text_loss


        #calculate pseudo label accuracy
        mask_label = labels > self.hparams.num_labeled_classes-1
        labels_unlab = labels[mask_label]
        for h in range(self.hparams.num_heads):
            text_pl = targets_text[h, ~mask_lab, nlc:]
            img_pl = targets_img[h, ~mask_lab, nlc:]
            pl = targets[h, ~mask_lab, nlc:]
            text_onehot_pl = text_pl.max(dim=-1)[1]
            img_onehot_pl = img_pl.max(dim=-1)[1]
            onehot_pl = pl.max(dim=-1)[1]
            if h == 0:
                img_pl_acc = cluster_acc(labels_unlab.detach().cpu().numpy(), img_onehot_pl.detach().cpu().numpy())
                text_pl_acc = cluster_acc(labels_unlab.detach().cpu().numpy(), text_onehot_pl.detach().cpu().numpy())
                pl_acc = cluster_acc(labels_unlab.detach().cpu().numpy(), onehot_pl.detach().cpu().numpy())
            elif h == self.hparams.num_heads-1:
                img_pl_acc += cluster_acc(labels_unlab.detach().cpu().numpy(), img_onehot_pl.detach().cpu().numpy())
                text_pl_acc += cluster_acc(labels_unlab.detach().cpu().numpy(), text_onehot_pl.detach().cpu().numpy())
                pl_acc += cluster_acc(labels_unlab.detach().cpu().numpy(), onehot_pl.detach().cpu().numpy())
                img_pl_acc = img_pl_acc/self.hparams.num_heads
                text_pl_acc = text_pl_acc/self.hparams.num_heads
                pl_acc = pl_acc / self.hparams.num_heads
            else:
                img_pl_acc += cluster_acc(labels_unlab.detach().cpu().numpy(), img_onehot_pl.detach().cpu().numpy())
                text_pl_acc += cluster_acc(labels_unlab.detach().cpu().numpy(), text_onehot_pl.detach().cpu().numpy())
                pl_acc += cluster_acc(labels_unlab.detach().cpu().numpy(), onehot_pl.detach().cpu().numpy())


        img_consistency = -img_consist.mean().item()
        text_consistency = -text_consist.mean().item()

        # log
        results = {
            "img_loss": img_loss.detach(),
            "img_loss_cluster": img_loss_cluster.mean(),
            "text_loss": text_loss.detach(),
            "text_loss_cluster": text_loss_cluster.mean(),
            "loss": loss.detach(),
            "img pl acc": img_pl_acc,
            "text pl acc": text_pl_acc,
            "img consistency": img_consistency,
            "text consistency": text_consistency,
            "lab img consistency": lab_img_consistency,
            "lab text consistency": lab_text_consistency,
            "pl acc":pl_acc,
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
        }

        self.log_dict(results, on_step=False, on_epoch=True, sync_dist=True)
        return loss



    def validation_step(self, batch, batch_idx, dl_idx):
        images, texts, labels = batch
        tag = self.trainer.datamodule.dataloader_mapping[dl_idx]

        # forward
        img_outputs, text_outputs = self(images, texts)

        if "unlab" in tag:  # use clustering head
            img_preds = img_outputs["img_logits_unlab"]
            img_preds_inc = torch.cat(
                [
                    img_outputs["img_logits_lab"].unsqueeze(0).expand(self.hparams.num_heads, -1, -1),
                    img_outputs["img_logits_unlab"],
                ],
                dim=-1,
            )
        else:  # use supervised classifier
            img_preds = img_outputs["img_logits_lab"]
            img_best_head = torch.argmin(self.img_loss_per_head)
            img_preds_inc = torch.cat(
                [img_outputs["img_logits_lab"], img_outputs["img_logits_unlab"][img_best_head]], dim=-1
            )
        img_preds = img_preds.max(dim=-1)[1]
        img_preds_inc = img_preds_inc.max(dim=-1)[1]

        self.metrics[dl_idx].update(img_preds, labels)
        self.metrics_inc[dl_idx].update(img_preds_inc, labels)

    #acc on image branch
    def validation_epoch_end(self, _):
        results = [m.compute() for m in self.metrics]
        results_inc = [m.compute() for m in self.metrics_inc]
        # log metrics
        for dl_idx, (result, result_inc) in enumerate(zip(results, results_inc)):
            prefix = self.trainer.datamodule.dataloader_mapping[dl_idx]
            prefix_inc = "incremental/" + prefix
            if "unlab" in prefix:
                for (metric, values), (_, values_inc) in zip(result.items(), result_inc.items()):
                    name = "/".join([prefix, metric])
                    name_inc = "/".join([prefix_inc, metric])
                    avg = torch.stack(values).mean()
                    avg_inc = torch.stack(values_inc).mean()
                    best = values[torch.argmin(self.img_loss_per_head)]
                    best_inc = values_inc[torch.argmin(self.img_loss_per_head)]
                    self.log(name + "/avg", avg, sync_dist=True)
                    self.log(name + "/best", best, sync_dist=True)
                    self.log(name_inc + "/avg", avg_inc, sync_dist=True)
                    self.log(name_inc + "/best", best_inc, sync_dist=True)
            else:
                self.log(prefix + "/acc", result)
                self.log(prefix_inc + "/acc", result_inc)


def main(args):
    dm = get_datamodule(args, "discover")

    run_name = "-".join(["discover", args.dataset, args.comment])
    wandb_logger = pl.loggers.WandbLogger(
        save_dir=args.log_dir,
        name=run_name,
        project=args.project,
        # project='try',
        entity=args.entity,
        offline=args.offline,
    )
    # logger = pl.loggers.TensorBoardLogger('/UNO-main/tb_logs/discover', name=run_name)

    model = Discoverer(**args.__dict__)
    trainer = pl.Trainer.from_argparse_args(args, gpus='0', logger=wandb_logger)
    trainer.fit(model, dm)


if __name__ == "__main__":
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    if not args.multicrop:
        args.num_small_crops = 0
    args.num_crops = args.num_large_crops + args.num_small_crops


    main(args)
