## TrOCR Formula Recognition

基于TrOCR + UniMER-1M数据集，训练一个小而美的公式识别数据集。

## 注意事项

使用transformers训练前，需要在`import torch`前，指定`CUDA_VISIBLE_DEVICES`，否则会卡住。

## data.zip

```bash
data
├── 2014
│   ├── formulaire039-equation072.jpg
│   ├── caption.txt
│   ├── formulaire039-equation073.jpg
│   └── formulaire039-equation074.jpg
└── train
    ├── formulaire039-equation072.jpg
    ├── caption.txt
    ├── formulaire039-equation073.jpg
    └── formulaire039-equation074.jpg
```

## Dataset

[UniMER_Dataset](https://huggingface.co/datasets/wanderkid/UniMER_Dataset)

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
