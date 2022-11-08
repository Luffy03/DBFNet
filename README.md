# Deep Bilateral Filtering Network (DBFNet)
Code for TIP 2022 paper, [**"Deep Bilateral Filtering Network for Point-Supervised Semantic Segmentation in Remote Sensing Images"**](https://ieeexplore.ieee.org/document/9745130), accepted.

Authors: Linshan Wu, <a href="https://www.leyuanfang.com/">Leyuan Fang</a>, Jun Yue, <a href="https://scholar.google.com/citations?user=dlZuABAAAAAJ&hl=en">Bob Zhang</a>, <a href="https://scholar.google.com/citations?user=Gr9afd0AAAAJ&hl=en">Pedram Ghamisi</a>, and Min He
## Getting Started

### Prepare Dataset
Download the Potsdam and Vaihingen dataset after processing [<b>weights</b>](https://drive.google.com/drive/folders/1oenWpYADqd-tTx7JeDQknxRNd3mgW2kQ)
```bash
ln -s </path/to/LoveDA> ./LoveDA
```


## Evaluate DCA Model on the test set
### 1. Download the pre-trained [<b>weights</b>](https://drive.google.com/drive/folders/1oenWpYADqd-tTx7JeDQknxRNd3mgW2kQ)
### 2. Move weight file to log directory
```bash
mkdir -vp ./log/
mv ./URBAN_0.4635.pth ./log/URBAN_0.4635.pth
mv ./RURAL_0.4517.pth ./log/RURAL_0.4517.pth
python My_test.py
```

### 3. Evaluate on the website
Submit your test results on [LoveDA Unsupervised Domain Adaptation Challenge](https://codalab.lisn.upsaclay.fr/competitions/424) and you will get your Test score.

Or you can download our [<b>results</b>](https://drive.google.com/drive/folders/1WQdgveVwW016BMKvw1Afj6o_MQ9UcZeA)
## Train DCA Model
```bash 
python DCA_train.py
```
The training [<b>logs</b>](https://drive.google.com/drive/folders/1oenWpYADqd-tTx7JeDQknxRNd3mgW2kQ)

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
