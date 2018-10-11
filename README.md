# 中英机器文本翻译

评测中英文本机器翻译的能力。机器翻译语言方向为中文到英文。


## 依赖

- Python 3.5
- PyTorch 0.4

## 数据集

我们使用AI Challenger 2017中的英中机器文本翻译数据集，超过1000万的英中对照的句子对作为数据集合。其中，训练集合占据绝大部分，验证集合8000对，测试集A 8000条，测试集B 8000条。

可以从这里下载：[英中翻译数据集](https://challenger.ai/datasets/translation)

![image](https://github.com/foamliu/Machine-Translation-v2/raw/master/images/dataset.png)

## 用法

### 数据预处理
提取训练和验证样本：
```bash
$ python pre-process.py
```

### 训练
```bash
$ python train.py
```


### Demo
下载 [预训练模型](https://github.com/foamliu/Scene-Classification/releases/download/v1.0/model.85-0.7657.hdf5) 放在 models 目录然后执行:

```bash
$ python demo.py
```