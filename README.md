# Deep_Rural_livelihood_model
# This work was supported by: *** and my love
================================================================
# Mapping China's rural livelihood index
<img width="1802" height="311" alt="image" src="https://github.com/user-attachments/assets/f1c883fa-2063-4bda-8cf0-fbda70da5644" />



This repository includes the code and data necessary to reproduce the results and figures for the article "Mapping China's rural livelihood" published in *########* on May 22, 2026 ([link](https://*****)).

Please cite this article as follows, or use the BibTeX entry below.

> Dawazhaxi, *et al*. Mapping China's rural livelihood. *####* **11**, 2583 (2026). https://*******

```tex
#####
```


## Hardware and Software Requirements

This code was tested on a system with the following specifications:

- operating system: Windows 10
- CPU: Intel(R) Xeon(R) Silver 4214 CPU @ 2.20GHz   2.19 GHz
- memory (RAM): 125GB
- disk storage: 500GB
- Using device: cuda
- GPU: Quadro P2200, Memory: 5.37 GB



The main software requirements are Python 3.13 with TensorFlow r1.15, and R 3.6. 

```bash
####requirements.txt#####
```

If you are using a GPU, you may need to also install CUDA 10 and cuDNN 7.


## Data Preparation Instructions
1. **Export satellite images from Google Earth Engine.** Follow the instructions in the `Export_images_from_GEE.js` notebook.
2. **Process the satellite images.** Follow the instructions in the `全国乡镇代码匹配到乡镇调查和街景数据.ipynb` and `全国乡镇街道办事处牧场等重分类为0或1.ipynb` notebooks. Then run the `样本扩充_分位数XGBoost回归.ipynb` notebooks.
3. **Prepare the data files.** Follow the instructions in the `提取乡_镇或街道或街道办事处或办事处V2.csv`, `全国乡镇.csv` and `indexes_of_30667_towns_with_natcodes_bycode_name.csv` notebooks in `Pre-Data`.
4. **Then created `Sample_2020.shp` in ArcGIS Pro uploade in the Data.


## Model Training Instructions
1. **Run the baseline linear models.** Follow the instructions in `models/dhs_baselines.ipynb`, `models/lsms_baselines.ipynb`, , and `models/lsmsdelta_baselines.ipynb`.
2. **Train the convolutional neural network models.** If running this code on a SLURM-enabled computing cluster, run the scripts `train_directly_runner.py` and `train_directly_lsm_runner.py`. Otherwise, run `train_directly.py` and `train_delta.py` with the desired command-line arguments to set hyperparameters.
3. **Extract learned feature representations.** Run the scripts `extract_features_dhs.py` and `extract_features_lsmsdelta.py`.
4. **Run cross-validated ridge-regression.** Follow the instructions in `models/dhs_ridge_resnet.ipynb` and `model_analysis/lsmsdelta_resnet.ipynb`.

# Model structure
```text
           Day images                         Night images
     (7 bands:RED, GREEN, BLUE,              Nighttime light
     NIR, SWIR1, SWIR2, TEMP1 )              (1 band: VIIRS)
               │                                     │
     ┌─────────┴───────────┐             ┌───────────┴─────────┐
     │ 1×1 Conv + BN + ReLU              │ 1×1 Conv + BN + ReLU
     │ (SourceAdapter)                   │    (SourceAdapter)
     └─────────┬───────────┘             └───────────┬─────────┘
               │                                     │
               └─────────── Concatenate ─────────────┘     
                   (2 × C_shared = 128 channels)
                                 │
   ┌─────────────────────────────┴──────────────────────────────┐
   │                 ConvStem (num_blocks=5)                    │
   │    = Conv → BN → ReLU → ResidualBlock×5 → Conv → BN → ReLU │
   └─────────────────────────────┬──────────────────────────────┘
                                 │
                   CompositionHead（Dirichlet α）
                  Conv → BN → ReLU → Dropout → Conv
                                 │
                           α → softplus + 1
                                 │
                   中心像元 Dirichlet 负对数似然 loss
