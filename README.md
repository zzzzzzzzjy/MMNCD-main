# Novel Class Discovery via Paired Images and Text
> **AAAI 2024**

> Paper: https://ojs.aaai.org/index.php/AAAI/article/view/28598

> **Abstract:** *In this paper, we propose a multi-modal novel class discovery method based on paired images and text, inspired by the low classification accuracy of chest X-ray images and the relatively higher accuracy of the paired text. Specifically, we first pretrain the image encoder and text encoder with multi-modal contrastive learning on the entire dataset and then we generate pseudo-labels separately on the image branch and text branch. We utilize intra-modal consistency to assess the quality of pseudo-labels and adjust the weights of the pseudo-labels from both branches to generate the ultimate pseudo-labels for training.*
<br>
<p align="center">
    <img src="./assets/UNO-teaser.png"/ width=50%> <br />
    <em>
    A visual comparison of our UNified Objective (UNO) with previous works.
    </em>
</p>
<br>
<p align="center">
    <img src="./assets/UNO-method.png"/ width=100%> <br />
    <em>
    Overview of the proposed architecture.
    </em>
</p>
<br>


# Installation
Our implementation is based on [UNO](https://github.com/DonkeyShot21/UNO), please follow the installation of their project.

# Datasets
We have provided examples of data in the demo.
For MIMIC-CXR-JPG Dataset, you can register and download according to the instructions on this [website](https://physionet.org/content/mimic-cxr-jpg/2.1.0/).
For Chest ImaGenome Dataset, you can download according to the instructions on this [website](https://physionet.org/content/chest-imagenome/1.0.0/).


# Commands
### Discovery
Running discovery on set1-1:
```
python main_discover.py --dataset MIMIC --max_epochs 200 --batch_size 128 --num_labeled_classes 4 --num_unlabeled_classes 3 --num_heads 4 --comment sub1-1 --precision 16 --multicrop
```

# Citation
If you like our work, please cite our paper:
```
@inproceedings{zhou2024novel,
  title={Novel Class Discovery in Chest X-rays via Paired Images and Text},
  author={Zhou, Jiaying and Liu, Yang and Chen, Qingchao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={7},
  pages={7650--7658},
  year={2024}
}
```
And please cite UNO:
```
@InProceedings{fini2021unified,
    author    = {Fini, Enrico and Sangineto, Enver and Lathuilière, Stéphane and Zhong, Zhun and Nabi, Moin and Ricci, Elisa},
    title     = {A Unified Objective for Novel Class Discovery},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2021}
}
```
