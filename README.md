# √ TrOCR Formula Recognition

❓缘由：看到[UniMERNet](https://github.com/opendatalab/UniMERNet)的工作，从他们发布的模型存储大小（4.91G）来看，实在太重了。同时，他们也发布了一个很大很全的公式识别数据集：UniMER_Dataset。

🎯 于是，想着基于TrOCR + UniMER-1M数据集，训练一个小而美的公式识别数据集。

仓库将UniMERNet作为Baseline，目标是超过UniMERNet，同时模型要小很多。

仓库dataset目录下为UniMER-1M的Tiny版，只用来测试程序使用。

### ⚠️注意事项

- 使用transformers训练前，需要在`import torch`前，指定`CUDA_VISIBLE_DEVICES`，否则会卡住。
- 以下实验数据，除**Exp1_1**外，其他的暂时都没有添加HME100K数据集
- 所有实验均采用`microsoft/trocr-small-stage1`作为预训练模型训练的。

#### TODO

- [ ] 给出速度基准
- [ ] 推理采用Flash Attention加速。（transformers==4.44.2中VisionEncoderDecoderModel不支持）
- [ ] 转ONNX模型，并比较推理速度
- [ ] 尝试使用xformers来优化推理速度

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
| Exp7 |  0.817  |   0.117  | 0.679  |  0.251     | 0.817  |  0.117   | 0.781  | 0.148  |
| Exp8 |  **0.886**  |   **0.07**  | **0.822**  |  **0.108**     | **0.633**  |  **0.217**   | **0.897**  | **0.07**  |
| Exp9 |  0.862  |   0.10  | 0.740  |  0.180     | 0.639  |  0.211   | 0.826  | 0.119  |
| Exp10 |  0.900  |   0.07  | 0.841  |  0.09     | 0.594  |  0.227   | 0.912  | 0.06  |

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
| Exp7  | 与Exp6相比，单一变量：增加HME100k数据集                                      |
| Exp8  | 与Exp7相比，单一变量：epoch=1 → epoch=10                                    |
| Exp9  | 与Exp8相比，单一变量：增加fusion-image-to-latex-datasets数据集（3069505）, Epoch=1       |
| Exp10  | 与Exp9相比，单一变量：epoch=1 → epoch=10 (fusion-image-to-latex-dataset 3467214)       |

### 🦩 Checkpoint

- [Exp5_1](https://huggingface.co/SWHL/TrOCR-Formula-Rec/tree/main/Exp5_1)
- [Exp8](https://huggingface.co/SWHL/TrOCR-Formula-Rec/tree/main/Exp8)
- [Exp10](https://huggingface.co/SWHL/TrOCR-Formula-Rec/tree/main/Exp10)

### 🔢 Dataset

⚠️注意：仓库中`dataset`目录下为示例，完整数据集需自行下载补充。

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

### 🔢 其他数据集和项目

- [fusion-image-to-latex-datasets](https://huggingface.co/datasets/hoang-quoc-trung/fusion-image-to-latex-datasets)
- [TexTeller](https://github.com/OleehyO/TexTeller)

### 📞 微信交流群

微信关注公众号：RapidAI, 后台回复“公式识别”即可进群

### 📚 Reference

- [UniMERNet](https://github.com/opendatalab/UniMERNet)
- [TrOCR-Handwritten-Mathematical-Expression-Recognition](https://github.com/win5923/TrOCR-Handwritten-Mathematical-Expression-Recognition.git)
- [Transformers-Tutorials](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb)
