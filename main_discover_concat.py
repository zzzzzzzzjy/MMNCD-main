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
from utils.nets import MMCXRNCD,MultiHeadConcat
from utils.eval import ClusterMetrics, cluster_acc
from utils.pseudo_label_generation import SinkhornKnopp, js_divergence

import numpy as np
from argparse import ArgumentParser
from datetime import datetime


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
        self.model = MultiHeadConcat(
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
        image_pretrained = torch.load('/MMNCD-main/pretrained/sub1_set4_resnet.pth')
        text_pretrained = torch.load('/MMNCD-main/pretrained/sub1_set4_bio.pth')

        self.model.img_encoder.load_state_dict(image_pretrained.state_dict(), strict=False)
        self.model.text_encoder.load_state_dict(text_pretrained.state_dict(), strict=False)

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
        self.register_buffer("loss_per_head", torch.zeros(self.hparams.num_heads))
        # self.register_buffer("text_loss_per_head", torch.zeros(self.hparams.num_heads))

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
        self.loss_per_head = torch.zeros_like(self.loss_per_head)

    def unpack_batch(self, batch):
        #no multiview
        images, texts, labels = batch
        mask_lab = labels < self.hparams.num_labeled_classes
        return images, texts, labels, mask_lab

    def training_step(self, batch, _):
        images, texts, labels, mask_lab = self.unpack_batch(batch)
        nlc = self.hparams.num_labeled_classes

        # normalize prototypes
        self.model.normalize_prototypes()
        # forward
        outputs = self.model(images, texts)

        # gather outputs
        outputs['img_text_logits_lab'] = (
            outputs['img_text_logits_lab'].unsqueeze(0).expand(self.hparams.num_heads, -1, -1)
        )

        logits = torch.cat([outputs['img_text_logits_lab'], outputs['img_text_logits_unlab']], dim=-1)
        logits_over = torch.cat([outputs['img_text_logits_lab'], outputs['img_text_logits_unlab_over']], dim=-1)


        # create targets
        targets_lab = (
            F.one_hot(labels[mask_lab], num_classes=self.hparams.num_labeled_classes)
            .float()
            .to(self.device)
        )
        targets = torch.zeros_like(logits)
        targets_over = torch.zeros_like(logits_over)



        # generate pseudo-labels with sinkhorn-knopp and fill unlab targets
        for h in range(self.hparams.num_heads):
            targets[h, mask_lab, :nlc] = targets_lab.type_as(targets)
            targets_over[h, mask_lab, :nlc] = targets_lab.type_as(targets)
            # distinguish labeled and unlabeled samples by label
            targets[h, ~mask_lab, nlc:] = self.sk(
                logits=outputs["img_text_logits_unlab"][h, ~mask_lab]
            ).type_as(targets)
            targets_over[h, ~mask_lab, nlc:] = self.sk(
                logits=outputs["img_text_logits_unlab_over"][h, ~mask_lab]
            ).type_as(targets_over)


        loss_cluster = self.cross_entropy_loss(logits, targets)
        loss_overcluster = self.cross_entropy_loss(logits_over, targets_over)

        # update best head tracker
        self.loss_per_head += loss_cluster.clone().detach()

        # total loss
        loss_cluster = loss_cluster.mean()
        loss_overcluster = loss_overcluster.mean()
        if torch.isnan(loss_overcluster).any():
            loss = loss_cluster
        else:
            loss = (loss_cluster+loss_overcluster)/2


        # log
        results = {
            "loss": loss.detach(),
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
        }

        self.log_dict(results, on_step=False, on_epoch=True, sync_dist=True)
        return loss



    def validation_step(self, batch, batch_idx, dl_idx):
        images, texts, labels = batch
        tag = self.trainer.datamodule.dataloader_mapping[dl_idx]

        # forward
        outputs = self(images, texts)

        if "unlab" in tag:  # use clustering head
            img_preds = outputs["img_text_logits_unlab"]
            img_preds_inc = torch.cat(
                [
                    outputs["img_text_logits_lab"].unsqueeze(0).expand(self.hparams.num_heads, -1, -1),
                    outputs["img_text_logits_unlab"],
                ],
                dim=-1,
            )
        else:  # use supervised classifier
            img_preds = outputs["img_text_logits_lab"]
            img_best_head = torch.argmin(self.loss_per_head)
            img_preds_inc = torch.cat(
                [outputs["img_text_logits_lab"], outputs["img_text_logits_unlab"][img_best_head]], dim=-1
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
                    best = values[torch.argmin(self.loss_per_head)]
                    best_inc = values_inc[torch.argmin(self.loss_per_head)]
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
        entity=args.entity,
        offline=args.offline,
    )
    # logger = pl.loggers.TensorBoardLogger('/UNO-main/tb_logs/discover', name=run_name)

    model = Discoverer(**args.__dict__)
    trainer = pl.Trainer.from_argparse_args(args, gpus='3', logger=wandb_logger)
    trainer.fit(model, dm)


if __name__ == "__main__":
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    if not args.multicrop:
        args.num_small_crops = 0
    args.num_crops = args.num_large_crops + args.num_small_crops

    main(args)
