import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoTokenizer, AutoModel


import torch
import torch.nn.functional as F
from torchvision import models


class Prototypes(nn.Module):
    def __init__(self, output_dim, num_prototypes):
        super().__init__()

        self.prototypes = nn.Linear(output_dim, num_prototypes, bias=False)

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.prototypes.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.prototypes.weight.copy_(w)

    def forward(self, x):
        return self.prototypes(x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=1):
        super().__init__()

        # layers = [
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(inplace=True),
        # ]
        # for _ in range(num_hidden_layers - 1):
        #     layers += [
        #         nn.Linear(hidden_dim, hidden_dim),
        #         nn.BatchNorm1d(hidden_dim),
        #         nn.ReLU(inplace=True),
        #     ]
        # layers.append(nn.Linear(hidden_dim, output_dim))
        layers = [nn.Linear(input_dim, output_dim),]
        self.mlp = nn.Sequential(*layers)       # linear(input-hidden), batchnorm1d, relu, linear(hidden-output)

    def forward(self, x):
        return self.mlp(x)



#num_heads个MLP和prototype层，MLP两个线性层：input-hidden-output; prototype: output-num_prototypes
#总结： 一个head就是一个MLP(双线性层) + 一个prototype(单线性层)
class MultiHead(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_prototypes, num_heads, num_hidden_layers=1
    ):
        super().__init__()
        self.num_heads = num_heads

        # projectors
        self.projectors = torch.nn.ModuleList(
            [MLP(input_dim, hidden_dim, output_dim, num_hidden_layers) for _ in range(num_heads)]
        )

        # prototypes
        self.prototypes = torch.nn.ModuleList(
            [Prototypes(output_dim, num_prototypes) for _ in range(num_heads)]
        )
        self.normalize_prototypes()

    @torch.no_grad()
    def normalize_prototypes(self):
        for p in self.prototypes:
            p.normalize_prototypes()

    def forward_head(self, head_idx, feats):
        z = self.projectors[head_idx](feats)
        z = F.normalize(z, dim=1)
        return self.prototypes[head_idx](z), z

    def forward(self, feats):
        out = [self.forward_head(h, feats) for h in range(self.num_heads)]
        return [torch.stack(o) for o in map(list, zip(*out))]



# model比较简单，首先一个 encoder，这里采用的resnet18, 我们的实验中替换成了resnet50
# pretrain stage: 只有 labeled head: 分类的线性层，
# discover stage: 分head_unlabel 和 head_unlabel_over
# head_unlabel包含 num_heads个head(MLP + prototype)
# head_unlabel_over 同样包含 num_heads个head(MLP+prototype), 不同的是 prototype的输出是 over_factor * num_unlabeled_classes

class MultiHeadResNet(nn.Module):
    def __init__(
        self,
        arch,
        low_res,
        num_labeled,
        num_unlabeled,
        hidden_dim=2048,
        proj_dim=256,
        overcluster_factor=3,
        num_heads=4,
        num_hidden_layers=1,
    ):
        super().__init__()

        # backbone
        self.img_encoder = models.__dict__[arch]()
        self.feat_dim = self.img_encoder.fc.weight.shape[1]
        self.img_encoder.fc = nn.Identity()
        # modify the encoder for lower resolution
        if low_res:
            self.img_encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.img_encoder.maxpool = nn.Identity()
            self._reinit_all_layers()

        self.img_head_lab = Prototypes(self.feat_dim, num_labeled)
        if num_heads is not None:
            self.img_head_unlab = MultiHead(
                input_dim=self.feat_dim,
                hidden_dim=hidden_dim,
                output_dim=proj_dim,
                num_prototypes=num_unlabeled,
                num_heads=num_heads,
                num_hidden_layers=num_hidden_layers,
            )
            self.img_head_unlab_over = MultiHead(
                input_dim=self.feat_dim,
                hidden_dim=hidden_dim,
                output_dim=proj_dim,
                num_prototypes=num_unlabeled * overcluster_factor,
                num_heads=num_heads,
                num_hidden_layers=num_hidden_layers,
            )

    @torch.no_grad()
    def _reinit_all_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def normalize_prototypes(self):
        self.img_head_lab.normalize_prototypes()
        if getattr(self, "img_head_unlab", False):
            self.img_head_unlab.normalize_prototypes()
            self.img_head_unlab_over.normalize_prototypes()

    def forward_heads(self, feats):
        out = {"img_logits_lab": self.img_head_lab(F.normalize(feats))}
        if hasattr(self, "img_head_unlab"):
            logits_unlab, proj_feats_unlab = self.img_head_unlab(feats)
            logits_unlab_over, proj_feats_unlab_over = self.img_head_unlab_over(feats)
            out.update(
                {
                    "img_logits_unlab": logits_unlab,
                    "img_proj_feats_unlab": proj_feats_unlab,
                    "img_logits_unlab_over": logits_unlab_over,
                    "img_proj_feats_unlab_over": proj_feats_unlab_over,
                }
            )
        return out

    def forward(self, views):
        # print(type(views))
        if isinstance(views, list):
            print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            feats = [self.img_encoder(view) for view in views]
            out = [self.forward_heads(f) for f in feats]
            out_dict = {"img_feats": torch.stack(feats)}
            for key in out[0].keys():
                out_dict[key] = torch.stack([o[key] for o in out])
            return out_dict
        else:
            feats = self.img_encoder(views)
            out = self.forward_heads(feats)
            out["img_feats"] = feats
            return out


class MultiHeadBERT(nn.Module):
    def __init__(
        self,
        num_labeled,
        num_unlabeled,
        hidden_dim=2048,
        proj_dim=256,
        overcluster_factor=3,
        num_heads=4,
        num_hidden_layers=1,
    ):
        super().__init__()

        # backbone
        # self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        # self.text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.text_encoder = AutoModel.from_pretrained("/MMUNO/bioclinicalbert")
        self.feat_dim = 768

        self.text_head_lab = Prototypes(self.feat_dim, num_labeled)
        if num_heads is not None:
            self.text_head_unlab = MultiHead(
                input_dim=self.feat_dim,
                hidden_dim=hidden_dim,
                output_dim=proj_dim,
                num_prototypes=num_unlabeled,
                num_heads=num_heads,
                num_hidden_layers=num_hidden_layers,
            )
            self.text_head_unlab_over = MultiHead(
                input_dim=self.feat_dim,
                hidden_dim=hidden_dim,
                output_dim=proj_dim,
                num_prototypes=num_unlabeled * overcluster_factor,
                num_heads=num_heads,
                num_hidden_layers=num_hidden_layers,
            )

    @torch.no_grad()
    def _reinit_all_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def normalize_prototypes(self):
        self.text_head_lab.normalize_prototypes()
        if getattr(self, "text_head_unlab", False):
            self.text_head_unlab.normalize_prototypes()
            self.text_head_unlab_over.normalize_prototypes()

    def forward_heads(self, feats):
        out = {"text_logits_lab": self.text_head_lab(F.normalize(feats))}
        if hasattr(self, "text_head_unlab"):
            logits_unlab, proj_feats_unlab = self.text_head_unlab(feats)
            logits_unlab_over, proj_feats_unlab_over = self.text_head_unlab_over(feats)
            out.update(
                {
                    "text_logits_unlab": logits_unlab,
                    "text_proj_feats_unlab": proj_feats_unlab,
                    "text_logits_unlab_over": logits_unlab_over,
                    "text_proj_feats_unlab_over": proj_feats_unlab_over,
                }
            )
        return out

    def forward(self, texts):
        xx, text_features = self.text_encoder(input_ids=texts['input_ids'].squeeze(1), attention_mask=texts['attention_mask'], return_dict=False)
        out = self.forward_heads(text_features)
        out["text_feats"] = text_features
        return out


class MultiHeadConcat(nn.Module):
    def __init__(
        self,
        arch,
        low_res,
        num_labeled,
        num_unlabeled,
        hidden_dim=2048,
        proj_dim=256,
        overcluster_factor=3,
        num_heads=4,
        num_hidden_layers=1,
    ):
        super().__init__()

        # backbone
        # self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        # self.text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.text_encoder = AutoModel.from_pretrained("/MMUNO/bioclinicalbert")
        self.feat_dim = 768 + 2048

        self.img_encoder = models.__dict__[arch]()
        # self.feat_dim = self.img_encoder.fc.weight.shape[1]
        self.img_encoder.fc = nn.Identity()
        # modify the encoder for lower resolution
        if low_res:
            self.img_encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.img_encoder.maxpool = nn.Identity()
            self._reinit_all_layers()

        self.img_text_head_lab = Prototypes(self.feat_dim, num_labeled)
        if num_heads is not None:
            self.img_text_head_unlab = MultiHead(
                input_dim=self.feat_dim,
                hidden_dim=hidden_dim,
                output_dim=proj_dim,
                num_prototypes=num_unlabeled,
                num_heads=num_heads,
                num_hidden_layers=num_hidden_layers,
            )
            self.img_text_head_unlab_over = MultiHead(
                input_dim=self.feat_dim,
                hidden_dim=hidden_dim,
                output_dim=proj_dim,
                num_prototypes=num_unlabeled * overcluster_factor,
                num_heads=num_heads,
                num_hidden_layers=num_hidden_layers,
            )

    @torch.no_grad()
    def _reinit_all_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def normalize_prototypes(self):
        self.img_text_head_lab.normalize_prototypes()
        if getattr(self, "img_text_head_unlab", False):
            self.img_text_head_unlab.normalize_prototypes()
            self.img_text_head_unlab_over.normalize_prototypes()

    def forward_heads(self, feats):
        out = {"img_text_logits_lab": self.img_text_head_lab(F.normalize(feats))}
        if hasattr(self, "img_text_head_unlab"):
            logits_unlab, proj_feats_unlab = self.img_text_head_unlab(feats)
            logits_unlab_over, proj_feats_unlab_over = self.img_text_head_unlab_over(feats)
            out.update(
                {
                    "img_text_logits_unlab": logits_unlab,
                    "img_text_proj_feats_unlab": proj_feats_unlab,
                    "img_text_logits_unlab_over": logits_unlab_over,
                    "img_text_proj_feats_unlab_over": proj_feats_unlab_over,
                }
            )
        return out

    def forward(self, imgs, texts):
        xx, text_features = self.text_encoder(input_ids=texts['input_ids'].squeeze(1), attention_mask=texts['attention_mask'], return_dict=False)
        img_feats = self.img_encoder(imgs)
        out = self.forward_heads(torch.cat((img_feats,text_features),dim=-1))
        out["img_feats"] = img_feats
        # out = self.forward_heads(text_features)
        out["text_feats"] = text_features
        return out


class MMCXRNCD(nn.Module):
    def __init__(
        self,
        arch,
        low_res,
        num_labeled,
        num_unlabeled,
        hidden_dim=2048,
        proj_dim=256,
        overcluster_factor=3,
        num_heads=4,
        num_hidden_layers=1,
    ):
        super().__init__()

        self.image_branch = MultiHeadResNet(arch=arch, low_res=low_res,num_labeled=num_labeled,
                                            num_unlabeled= num_unlabeled, hidden_dim=hidden_dim,
                                            proj_dim=proj_dim, overcluster_factor=overcluster_factor,
                                            num_heads=num_heads, num_hidden_layers=num_hidden_layers)


        self.text_branch = MultiHeadBERT(num_labeled=num_labeled, num_unlabeled=num_unlabeled,
                                         hidden_dim=hidden_dim, proj_dim=proj_dim,
                                         overcluster_factor=overcluster_factor, num_heads=num_heads,
                                         num_hidden_layers=num_hidden_layers)


    def forward(self, images, texts):
        img_out = self.image_branch(images)
        text_out = self.text_branch(texts)
        return img_out, text_out