# 🧮 TrOCR Formula Recognition

❓缘由：看到[UniMERNet](https://github.com/opendatalab/UniMERNet)的工作，从他们发布的模型存储大小（4.91G）来看，实在太重了。同时，他们也发布了一个很大很全的公式识别数据集：UniMER_Dataset。

🎯 于是，想着基于TrOCR + UniMER-1M数据集，训练一个小而美的公式识别数据集。

仓库将UniMERNet作为Baseline，目标是超过UniMERNet，同时模型要小很多。

仓库dataset目录下为UniMER-1M的Tiny版，只用来测试程序使用。

### ⚠️注意事项

- 使用transformers训练前，需要在`import torch`前，指定`CUDA_VISIBLE_DEVICES`，否则会卡住。
- 以下实验数据，除**Exp1_1**外，其他的暂时都没有添加HME100K数据集

### 🔬 实验记录

实验表格来自[UniMERNet](https://arxiv.org/abs/2404.15254) Table 5

| Method   | SPE-BLEU↑ | SPE-EditDis↓ | CPE-BLEU↑ | CPE-EditDis↓ | SCE-BLEU↑ | SCE-EditDis↓ | HWE-BLEU↑ | HWE-EditDis↓ |
| :---- | :-------: | :----------: | :-------: | :----------: | :-------: | :----------: | :-------: | :----------: |
| [Pix2tex](https://github.com/lukas-blecher/LaTeX-OCR) |   0.873   |    0.088     |   0.655   |    0.408     |   0.092   |    0.817     |   0.012   |    0.920     |
| [Texify](https://github.com/VikParuchuri/texify)      |   0.906   |    0.061     |   0.690   |    0.230     |   0.420   |    0.390     |   0.341   |    0.522     |
| [UniMERNet](https://github.com/opendatalab/UniMERNet) |   0.917   |    0.058     |   0.916   |    0.060     |   0.616   |    0.229     |   0.921   |    0.055     |
|||||||||
| Exp1   |   0.815   |    0.121     |   0.677   |    0.259     |   0.589   |    0.227     |   0.150   |    0.520     |
| Exp1_1 |   0.883   |    0.07     |   0.810   |    0.122     |   0.489   |    0.262     |   0.900   |    0.06     |
| Exp2    |   0.798   |    0.132     |   0.677   |    0.259     |   0.589   |    0.227     |   0.150   |    0.520     |
| Exp3 |   0.813   |    0.127     |   0.682   |    0.263     |   0.302   |   0.231     |   0.166   |   0.540      |
| Exp4 |   0.873   |   0.077    |  0.801   |   0.130     |   0.550  |   0.238    |  0.092   |   0.469     |
| Exp5 |  0.846  |   0.201  | 0.823  |  0.134     | 0.418  |  0.553   | 0.05  | 0.6724  |
| Exp5_1 |  0.819  |   0.119  | 0.682  |  0.249     | 0.595  |  0.230   | 0.179  | 0.512  |
| Exp6 |  0.812  |   0.116  | 0.676  |  0.253     | 0.657  |  0.210   | 0.342  | 0.404  |

|  Exp  | 说明                                                                                                   |
| :--- | :----------------------------------------------------------------------------------------------------- |
| Exp1  | 首次基于UniMER-1M训练，采用预训练模型是`microsoft/trocr-small-stage1` <br/> 采用TrOCR默认Tokenizer |
| Exp1_1 | 基于Exp1，控制单一变量：训练30个Epoch by [limaopeng1](https://github.com/limaopeng1) |
| Exp2  | 更改LaTex-OCR方法用的BPE Tokenizer                                                                   |
| Exp3  | 修复Exp2中model配置bug                                                                               |
| Exp4  | 与Exp3相比，单一变量：epoch=1 → epoch=5                                                             |
| Exp5  | 与Exp1相比，单一变量：epoch=1 → epoch=10                                                             |
| Exp5_1  | 补充实验，修复Exp5中，去掉text前后加了BOS和EOS的地方，只跑一个epoch                                            |
| Exp6  | 与Exp5_1相比，单一变量：参考UniMERNet源码，增加数据增强                                      |

### 🦩 Checkpoint

🔥 [Hugging Face](https://huggingface.co/SWHL/TrOCR-Formula-Rec)


### 🔢 Dataset

[UniMER_Dataset](https://huggingface.co/datasets/wanderkid/UniMER_Dataset)
完整的UniMER目录结构如下：

```text
dataset
├── UniMER-1M
│   ├── images
│   └── train.txt
└── UniMER-Test
    ├── cpe
    ├── hwe
    ├── sce
    ├── spe
    ├── cpe.txt
    ├── hwe.txt
    ├── sce.txt
    └── spe.txt
```

训练集总共1061,791 LaTeX-Image pairs。

测试集由4种类型公式组成，总共23757张图像：

- Simple Printed Expressions (SPE): 6,762 samples
- Complex Printed Expressions (CPE): 5,921 samples
- Screen Capture Expressions (SCE): 4,742 samples
- Handwritten Expressions (HWE): 6,332 samples

各个种类示例图像如下：

<div align="center">
    <img src="https://github.com/SWHL/TrOCR-Formula-Rec/releases/download/v0.0.0/dataset_deom.png">
</div>

### 📚 Reference

- [UniMERNet](https://github.com/opendatalab/UniMERNet)
- [TrOCR-Handwritten-Mathematical-Expression-Recognition](https://github.com/win5923/TrOCR-Handwritten-Mathematical-Expression-Recognition.git)
- [Transformers-Tutorials](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb)
