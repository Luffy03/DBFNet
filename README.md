# Deep Bilateral Filtering Network (DBFNet)
Code for TIP 2022 paper, [**"Deep Bilateral Filtering Network for Point-Supervised Semantic Segmentation in Remote Sensing Images"**](https://ieeexplore.ieee.org/document/9961229), accepted.

Authors: Linshan Wu, <a href="https://scholar.google.com/citations?hl=en&user=Gfa4nasAAAAJ">Leyuan Fang</a>, <a href="https://scholar.google.com/citations?user=epXQ1RwAAAAJ&hl=en&oi=ao">Jun Yue</a>, <a href="https://scholar.google.com/citations?user=dlZuABAAAAAJ&hl=en">Bob Zhang</a>, <a href="https://scholar.google.com/citations?user=Gr9afd0AAAAJ&hl=en">Pedram Ghamisi</a>, and Min He

## Getting Started
### Prepare Dataset
Download the Potsdam and Vaihingen [<b>datasets</b>](https://drive.google.com/drive/folders/1CiYzJyBn1rV-xsrsYQ6o2HDQjdfnadHl) after processing.

Or you can download the datasets from the official [<b>website</b>](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx). Then, crop the original images and create point labels following our code in [<b>Dataprocess</b>](https://github.com/Luffy03/DBFNet/tree/master/DataProcess).

If your want to run our code on your own datasets, the pre-process code is also available in [<b>Dataprocess</b>](https://github.com/Luffy03/DBFNet/tree/master/DataProcess).

## Evaluate
### 1. Download the [<b>original datasets</b>](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx)
### 2. Download our [<b>weights</b>](https://drive.google.com/drive/folders/1CiYzJyBn1rV-xsrsYQ6o2HDQjdfnadHl)
### 3. Run our code
```bash
python predict.py
```

## Train 
### 1. Train DBFNet
```bash 
python run/point/p_train.py
```
### 2. Generate pseudo labels
```bash 
python run/point/p_predict_train.py
```
### 1. Recursive learning
```bash 
python run/second/sec_train.py
```

## Citation ✏️ 📄

If you find this repo useful for your research, please consider citing the paper as follows:

```
@ARTICLE{Wu_DBFNet,
  author={Wu, Linshan and Fang, Leyuan and Yue, Jun and Zhang, Bob and Ghamisi, Pedram and He, Min},
  journal={IEEE Transactions on Image Processing}, 
  title={Deep Bilateral Filtering Network for Point-Supervised Semantic Segmentation in Remote Sensing Images}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TIP.2022.3222904}}
```

For any question, please contact [Linshan Wu](mailto:15274891948@163.com).
