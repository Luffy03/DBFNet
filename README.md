# Deep Bilateral Filtering Network (DBFNet)
Code for TIP 2022 paper, [**"Deep Bilateral Filtering Network for Point-Supervised Semantic Segmentation in Remote Sensing Images"**](https://ieeexplore.ieee.org/document/9745130), accepted.

Authors: Linshan Wu, <a href="https://www.leyuanfang.com/">Leyuan Fang</a>, Jun Yue, <a href="https://scholar.google.com/citations?user=dlZuABAAAAAJ&hl=en">Bob Zhang</a>, <a href="https://scholar.google.com/citations?user=Gr9afd0AAAAJ&hl=en">Pedram Ghamisi</a>, and Min He

## Getting Started

### Prepare Dataset
Download the Potsdam and Vaihingen [<b>datasets</b>](https://drive.google.com/drive/folders/1oenWpYADqd-tTx7JeDQknxRNd3mgW2kQ) after processing.

Or you can download the datasets from the official [<b>website</b>](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx). Then, crop the original images and create point labels following our code in [<b>Dataprocess</b>](https://github.com/Luffy03/DBFNet/tree/master/DataProcess).

If your want to run our code on your own datasets, the pre-process code is also available in [<b>Dataprocess</b>](https://github.com/Luffy03/DBFNet/tree/master/DataProcess).

## Evaluate DBFNet on the test set
### 1. Download our [<b>weights</b>](https://drive.google.com/drive/folders/1oenWpYADqd-tTx7JeDQknxRNd3mgW2kQ)
### 2. Run our code
```bash
python predict.py
```

## Train DBFNet
```bash 
python run/point/p_train.py
```

## Citation ✏️ 📄

If you find this repo useful for your research, please consider citing the paper as follows:

```
@ARTICLE{9745130,
  @ARTICLE{9745130,
  author={Wu, Linshan and Lu, Ming and Fang, Leyuan},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Deep Covariance Alignment for Domain Adaptive Remote Sensing Image Segmentation}, 
  year={2022},
  volume={60},
  number={},
  pages={1-11},
  doi={10.1109/TGRS.2022.3163278}}
```

For any questions, please contact [Linshan Wu](mailto:15274891948@163.com).
