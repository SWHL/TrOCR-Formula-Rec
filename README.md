## TrOCR Formula Recognition

❓缘由：看到[UniMERNet](https://github.com/opendatalab/UniMERNet)的工作，从他们发布的模型存储大小（4.91G）来看，实在太重了。同时，他们也发布了一个很大很全的公式识别数据集：UniMER_Dataset。

🎯 于是，想着基于TrOCR + UniMER-1M数据集，训练一个小而美的公式识别数据集。

仓库将UniMERNet作为Baseline，目标是超过UniMERNet，同时模型要小很多。

仓库dataset目录下为UniMER-1M的Tiny版，只用来测试程序使用。

### 🔬实验记录

实验表格来自[UniMERNet](https://arxiv.org/abs/2404.15254) Table 5

|Method|SPE-BLEU↑|SPE-EditDis↓|CPE-BLEU↑|CPE-EditDis↓|SCE-BLEU↑|SCE-EditDis↓|HWE-BLEU↑|HWE-EditDis↓|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[Pix2tex](https://github.com/lukas-blecher/LaTeX-OCR)|0.873|0.088|0.655|0.408|0.092|0.817|0.012|0.920|
|[Texify](https://github.com/VikParuchuri/texify)|0.906|0.061|0.690|0.230|0.420|0.390|0.341|0.522|
|[UniMERNet](https://github.com/opendatalab/UniMERNet)|0.917|0.058|0.916|0.060|0.616|0.229|0.921|0.055|
|Exp1|||||||||

#### Exp1 2024-07-23

- 首次基于UniMER-1M训练，采用预训练模型是`microsoft/trocr-small-stage1`

### ⚠️注意事项

使用transformers训练前，需要在`import torch`前，指定`CUDA_VISIBLE_DEVICES`，否则会卡住。

### Dataset

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

### Reference

- [UniMERNet](https://github.com/opendatalab/UniMERNet)
- [TrOCR-Handwritten-Mathematical-Expression-Recognition](https://github.com/win5923/TrOCR-Handwritten-Mathematical-Expression-Recognition.git)
- [Transformers-Tutorials](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb)
