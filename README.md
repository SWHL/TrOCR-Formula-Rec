## TrOCR Formula Recognition

基于TrOCR + UniMER-1M数据集，训练一个小而美的公式识别数据集。

仓库dataset目录下为UniMER-1M的Tiny版，只用来测试程序使用。

## 注意事项

使用transformers训练前，需要在`import torch`前，指定`CUDA_VISIBLE_DEVICES`，否则会卡住。

## Dataset

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

## Reference

- [TrOCR-Handwritten-Mathematical-Expression-Recognition](https://github.com/win5923/TrOCR-Handwritten-Mathematical-Expression-Recognition.git)
- [Transformers-Tutorials](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb)
