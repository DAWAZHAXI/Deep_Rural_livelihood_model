# Deep rural livelihoods model (DRLM) 🌾🏕💰🐏🐂🐖🐕🌳


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Paper](Currently writing)](https://*****)

> **Mapping China's Rural Livelihood Index using Multi-Modal Remote Sensing and Deep Learning**

This work was supported by: *****

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Results](#-results)
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Model Training](#-model-training)
- [Inference](#-inference-and-mapping)
- [Citation](#-citation)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Overview

<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/758b0b8c-1cb3-4b8e-b418-8bea4877c378" />

*Figure 1: Contribution of this study*

<img width="500" height="600" alt="image" src="https://github.com/user-attachments/assets/3628f0bb-d016-4d24-9183-6fc9c5ae9c27" />

*Figure 2: Mapping flow*

<img width="500" height="550" alt="image" src="https://github.com/user-attachments/assets/ab54e17d-4cb4-4025-b77c-aaf90a58ddfb" />

*Figure 3: Spatiotemporal distribution of rural livelihood index across China (2010-2020)*

This repository contains the complete implementation of our deep learning framework for mapping China's rural livelihood patterns using multi-modal satellite imagery. 

### 🎯 Research Scope

- **Spatial Coverage**: 30,667 rural townships across China
- **Temporal Range**: 2010-2020 (two year)
- **Spatial Resolution**: 30 meters
- **Prediction Accuracy**: R² = 0.75-0.85 across all components

### 🛰️ Data Sources

Our approach integrates:
- **Daytime Landsat-8/9** imagery (7 spectral bands: RED, GREEN, BLUE, NIR, SWIR1, SWIR2, THERMAL)
- **Nighttime VIIRS-DNB** data for socioeconomic indicators
- **Survey data** from 30,000+ rural townships

---

## ✨ Key Features

### 🔬 Technical Innovations

✅ **Dual-Branch Architecture**
- Separate feature extraction for day/night modalities
- Mid-level fusion for optimal information integration  
- Lightweight design (~1.2M parameters)

✅ **Dirichlet Distribution Output**
- Natural probability constraints (Σp = 1)
- Uncertainty quantification
- Theoretically principled framework

✅ **Multi-Scale Feature Learning**
- PreActivation ResNet blocks
- Deep residual connections (11 total blocks)
- No spatial downsampling (preserves 64×64 resolution)

✅ **Robust Training Strategy**
- 5-fold cross-validation
- Data augmentation (flips, rotations, brightness/contrast)
- Early stopping with patience
- AdamW optimizer with weight decay

### 🎯 Applications

- **Rural Development Planning**: Identify areas requiring targeted interventions
- **Poverty Alleviation**: Track socioeconomic changes over time
- **Environmental Monitoring**: Assess forest-agriculture transitions
- **Policy Evaluation**: Quantify impacts of rural revitalization programs

---

## 🏗️ Architecture

### Model Overview

Our **DualResNet-Dirichlet** model employs a dual-branch architecture for processing multi-modal satellite data:
<img width="1042" height="528" alt="image" src="https://github.com/user-attachments/assets/67ac3050-e451-4bb6-8846-f7c97bd78c49" />


### Component Definitions

Our model predicts four distinct rural livelihood types based on household income composition:

| Component | Abbreviation | Description | Income Structure |
|-----------|--------------|-------------|------------------|
| **Farming-dominant** | F | Household income is entirely or predominantly dependent on agricultural production activities | Primary: Agriculture (>70%)<br>Secondary: Minimal non-farm |
| **Farming with secondary non-farming income** | F_NF | Household income primarily derived from agriculture but supplemented by non-farming activities such as handicrafts, seasonal labor, or small business | Primary: Agriculture (50-70%)<br>Secondary: Non-farm income |
| **Non-farming with secondary farming income** | NF_F | Household income primarily from non-agricultural sources (manufacturing, services, trade) while maintaining subsistence or small-scale farming | Primary: Non-farm (50-70%)<br>Secondary: Agriculture |
| **Non-farming-dominant** | NF | Household income is entirely or predominantly derived from non-agricultural employment or business activities | Primary: Non-farm (>70%)<br>Secondary: Minimal agriculture |

**Mathematical Constraint**: F + F_NF + NF_F + NF = 1.0 (proportions sum to unity)

#### Livelihood Transition Spectrum
```
Pure Farming (F) ←─→ F_NF ←─→ NF_F ←─→ Pure Non-Farming (NF)
    │                                          │
    └──────── Transition Zone ─────────────────┘
```

**Real-World Examples:**
- **F**: Traditional grain-producing villages in Henan Province
- **F_NF**: Rice farming areas with emerging agritourism in Jiangxi
- **NF_F**: Peri-urban townships near Shenzhen with factory workers maintaining kitchen gardens
- **NF**: Fully industrialized townships in Dongguan manufacturing belt
**Constraint**: F + F_NF + NF_F + NF = 1.0

### Model Statistics

| Metric | Value |
|--------|-------|
| **Total Parameters** | 1,200,000 (1.2M) |
| **Model Size** | 4.8 MB (FP32) |
| **Input Shape** | Day: [B,7,64,64] + Night: [B,1,64,64] |
| **Output Shape** | [B, 4, 64, 64] |
| **Training Time** | ~5 h/fold (GPU) |
| **Inference Time** | ~10 min (10k×10k image, GPU) |

---

## 📊 Results

### Performance Metrics
<img width="500" height="450" alt="image" src="https://github.com/user-attachments/assets/dbd12cbf-03f3-4943-bfe8-e76bf4767bc0" />


### Validation Studies
<img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/d83a5450-5863-4436-bded-ef2cf1723d44" />

---

## 💻 Installation

### Hardware Requirements

| Component | Specification |
|-----------|---------------|
| **OS** | Windows 10 / Linux (Ubuntu 20.04+) |
| **CPU** | Intel Xeon W-2295|
| **RAM** | 128 GB (minimum 64 GB recommended) |
| **Storage** | 500 GB free space |
| **GPU** | NVIDIA RTX A5000 (24GB VRAM) (minimum 10 GB recommended) |
| **CUDA** | 11.0+ |
| **cuDNN** | 8.0+ |

### Quick Start

#### 1. Clone Repository

```bash
git clone https://github.com/DAWAZHAXI/Deep_Rural_livelihood_model.git
cd Deep_Rural_livelihood_model
```

#### 2. Create Environment

**Using conda (recommended):**
```bash
conda create -n rural_livelihood python=3.13
conda activate rural_livelihood
```

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```txt
# Core
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Geospatial
rasterio>=1.3.0
geopandas>=0.13.0
shapely>=2.0.0

# Processing
opencv-python>=4.8.0
tqdm>=4.65.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# GEE
earthengine-api>=0.1.360

# ML
xgboost>=2.0.0
```

#### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📂 Data Preparation

### Overview

Four main steps:
1. Export satellite imagery from Google Earth Engine
2. Process township-level data
3. Generate training samples via XGBoost
4. Create final Shapefile dataset

### Step 1: Export from GEE

Run the `Export_images_from_GEE.js` script in the Google Earth Engine code editor, and use '特征影像z_score处理.ipynb' to complete the normalised z-score with a mean of 0 and a standard deviation of 1.

**Output:**
```
exported_images/
├── Landsat_RED_2020_90m_zscore.tif
├── Landsat_GREEN_2020_90m_zscore.tif
├── Landsat_BLUE_2020_90m_zscore.tif
├── Landsat_NIR_2020_90m_zscore.tif
├── Landsat_SWIR1_2020_90m_zscore.tif
├── Landsat_SWIR2_2020_90m_zscore.tif
├── Landsat_TEMP1_2020_90m_zscore.tif
└── VIIRS_2020_90m_zscore.tif
```

### Step 2: Process Township Data

**Notebooks:**
- `全国乡镇代码匹配到乡镇调查和街景数据.ipynb`: Match codes
- `全国乡镇街道办事处牧场等重分类为0或1.ipynb`: Reclassify types

**Input files:**
- `提取乡_镇或街道或街道办事处或办事处V2.csv`
- `全国乡镇.csv`
- `indexes_of_30667_towns_with_natcodes_bycode_name.csv`

### Step 3: Sample Augmentation

**Notebook:** `样本扩充_分位数XGBoost回归.ipynb`

Expands samples from ~5,000 → ~30,000 using quantile regression.

### Step 4: Create Shapefile

Use ArcGIS Pro to create `Sample_2020.shp`:

```
Attributes:
├── ID: Unique identifier
├── F: [0-1]
├── F_NF: [0-1]
├── NF_F: [0-1]
├── NF: [0-1]
├── longitude: X coordinate
├── latitude: Y coordinate

Constraint: F + F_NF + NF_F + NF = 1.0
```
---

## 🚀 Model Training

### Configuration

**Key Hyperparameters:**
```python
PATCH_SIZE = 64         # Input size
BATCH_SIZE = 256        # Training batch
EPOCHS = 300            # Max epochs
PATIENCE = 15           # Early stopping
LEARNING_RATE = 3e-4    # Initial LR
WEIGHT_DECAY = 1e-3     # L2 regularization
```

### Training Commands

**Basic training:**
```bash
python Scripts/00.Train_complete_Random-5-Fold-CV.py\
    --data_dir Data/ \
    --output_dir model_outputs_2020/ \
    --batch_size 256 \
    --epochs 300 \
    --device cuda
```

**Quick test:**
```bash
python Scripts/00.Train_complete_Random-5-Fold-CV.py \
    --quick_test \
    --max_samples 500 \
    --epochs 20
```

### Training Pipeline

5-fold cross-validation:
```
Dataset (N=30,000)
    ├─ Fold 1: Train=24k | Val=3k | Test=3k
    ├─ Fold 2: Train=24k | Val=3k | Test=3k
    ├─ Fold 3: Train=24k | Val=3k | Test=3k
    ├─ Fold 4: Train=24k | Val=3k | Test=3k
    └─ Fold 5: Train=24k | Val=3k | Test=3k
```

### Output Files

```
model_outputs_2020/
├── model_fold1_rep0_lr0.001_wd0.01.pth
├── model_fold2_rep0_lr0.001_wd0.01.pth
├── ... (5 models)
├── stage2_fraction_results.csv
└── summary_report.txt
```

---

## 🗺️ Inference and Mapping

### Full-Image Inference

```bash
python Scripts/02.Inference_2020_optimized_final.py \
    --model_path model_outputs_2020/model_fold3_rep0_lr0.001_wd0.01.pth \
    --day_images Data/Landsat_NL_Mector_90m_zscore/ \
    --night_image Data/Landsat_NL_Mector_90m_zscore/VIIRS_2020_90m_zscore.tif \
    --output_dir predictions_2020/ \
    --patch_size 64 \
    --step 32 \
    --batch_size 16
```

**Key parameters:**
- `--patch_size 64`: Must match training
- `--step 32`: 50% overlap
- `--batch_size 16`: Adjust for GPU memory

### Output Products

```
predictions_2020/
├── pred_best_F_2020_90m.tif       
├── pred_best_F_NF_2020_90m.tif   
├── pred_best_NF_F_2020_90m.tif    
├── pred_best_NF_2020_90m.tif      
└── prediction_overview.png         # Visualization
```

### Visualization

```python
import rasterio
import matplotlib.pyplot as plt

with rasterio.open('predictions_2020/pred_best_F_2020_90m.tif') as src:
    forest = src.read(1)

plt.imshow(forest, cmap='YlGn', vmin=0, vmax=1)
plt.colorbar(label='Forest Proportion')
plt.title('Forest Component')
plt.savefig('forest_map.png', dpi=300)
```

---

## 📖 Citation

If you use this code in your research, please cite:

> Dawazhaxi, *et al*. "Mapping China's Rural Livelihood Index using Multi-Modal Remote Sensing and Deep Learning." *Journal Name* **11**, 2583 (2026). https://doi.org/*****

**BibTeX:**
```bibtex
@article{dawazhaxi2026rural,
  title={Mapping China's Rural Livelihood Index using Multi-Modal Remote Sensing and Deep Learning},
  author={Dawazhaxi and [Co-authors]},
  journal={Journal Name},
  volume={11},
  pages={2583},
  year={2026},
  doi={10.XXXX/XXXXX}
}
```

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

**Guidelines:**
- Follow PEP 8 style
- Add unit tests
- Update documentation

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

**Funding:**
- National Natural Science Foundation of China
- Chinese Academy of Sciences

**Data Sources:**
- Landsat-8: USGS/NASA LP DAAC
- VIIRS-DNB: NOAA NCEI
- Township Data: National Bureau of Statistics of China

**Tools:**
- Google Earth Engine
- PyTorch
- Rasterio & GeoPandas

---

## 📞 Contact

**Lead Author**: Dawazhaxi  
**GitHub**: [@DAWAZHAXI](https://github.com/DAWAZHAXI)  
**Email**: [15687851457@163.com](mailto:your.email@institution.edu)

**Report Issues:** [GitHub Issues](https://github.com/DAWAZHAXI/Deep_Rural_livelihood_model/issues)

---

<p align="center">
  <sub>Built with ❤️ for rural development research</sub>
</p>

<p align="center">
  <a href="#-overview">Back to Top ↑</a>
</p>
