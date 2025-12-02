# Deep rural livelihoods model (DRLM) 🌾🏕💰🐏🐂🐖


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Paper](Currently writing)](https://*****)

> **Mapping rural livelihood strategies to reveal the equality of urbanity**>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#️-architecture)
- [Results](#-results)
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Model Training](#-model-training)
- [Inference](#️-inference-and-mapping)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)
---

## 🌟 Overview

<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/758b0b8c-1cb3-4b8e-b418-8bea4877c378" />

*Figure 1: Contribution of this study*

<img width="500" height="600" alt="image" src="https://github.com/user-attachments/assets/3628f0bb-d016-4d24-9183-6fc9c5ae9c27" />

*Figure 2: Mapping flow*

<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/7a8b628c-a94b-473a-ad59-4f34acb56731" />

*Figure 3: Proportion of rural livelihoods dependent purely on farming (Spatial resolution: 90m)*

<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/d9af3814-3002-402a-97a5-d9683998a334" />

*Figure 4: Proportion of rural livelihoods primarily farming with non-farm as a secondary activity (spatial resolution: 90 metres)*

<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/3002c2ba-803b-49ba-a1f7-1eaf2c3c5118" />

*Figure 5: Proportion of rural livelihoods primarily non-farm with farming as a secondary activity (spatial resolution: 90 metres)*

<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/8585603e-cf6d-419c-87be-dcc6cac92953" />

*Figure 6: Proportion of rural livelihoods dependent purely on non-farm (Spatial resolution: 90m)*

This repository contains the complete implementation of our deep learning framework for mapping China's rural livelihood patterns using multi-modal satellite imagery. 

### 🎯 Research Scope

- **Spatial Coverage**: Rural settlements across China
- **Temporal Range**: 2010-2020 (two year)
- **Spatial Resolution**: 90 meters
- **Prediction Accuracy**: R² = 0.75-0.85 across all components

### 🛰️ Data Sources

Our approach integrates:
- **Daytime Landsat-8/9** imagery (7 spectral bands: RED, GREEN, BLUE, NIR, SWIR1, SWIR2, THERMAL)
- **Nighttime VIIRS-DNB** data (1 band: VIIRS)
- **Survey data** from 355 rural survey points and 30,000+ rural townships

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

**Real-World Examples:**
- **F**: Traditional grain-producing villages in Henan Province
- **F_NF**: Rice farming areas with emerging agritourism in Jiangxi
- **NF_F**: Peri-urban townships near Shenzhen with factory workers maintaining kitchen gardens
- **NF**: Fully industrialized townships in Dongguan manufacturing belt
**Constraint**: F + F_NF + NF_F + NF = 1.0

### Model Statistics
| Metric | Value |
|--------|-------|
| **Output Shape** | [B, 4, 64, 64] |
| **Training Time** | ~90 h/Random 5-fold and OOR model (GPU) |
| **Inference Time** | ~8 h (10k×10k image, GPU) |

---

## 📊 Results

### Performance Metrics
<img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/f5c04f78-3531-42da-a798-f8ce4bc857cb" />


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

### Stage 1: Random Cross-Validation → for global evaluation

**Script:** `00.Train_complete_Random-5-Fold-CV.py`

Trains global baseline model with 5-fold cross-validation.
```bash
# Full training
python Model/00.Train_complete_Random-5-Fold-CV.py

# Quick test (edit QUICK_TEST = True in script)
python Model/00.Train_complete_Random-5-Fold-CV.py
```

**Outputs:**
- 5 trained models: `model_fold{1-5}_rep0_lr0.0003_wd0.001.pth`
- Performance results: `stage2_fraction_results.csv`
- Visualizations: Run `01.Plot_results.ipynb`

**Expected R²:** 0.85-0.90 (100% data)  
---

### Stage 2: Regional Models → for partition weighting

**Script:** `00.Train_Out-of-Region_5-fold-CV.py`

Trains 6 region-specific models for spatial heterogeneity.

**Regions:** 东北, 华北, 华东, 中南, 西南, 西北
```bash
python Model/00.Train_Out-of-Region_5-fold-CV.py
```

**Strategy:** Each region uses 20% samples for testing, 80% + other regions for training.

**Outputs:** 6 regional models (`model_OOR_macro_soft_fold{1-6}_lr0.0003_wd0.001.pth`)

**Auto-Resume:** Re-run if interrupted - continues from checkpoint automatically.

---

## 🗺️ Inference and Mapping

**Script:** `02.inference_2020_random_add_region_全像元.py`

Generates national-scale livelihood maps using adaptive regional ensemble.
```bash
python Model/02.inference_2020_random_add_region_全像元.py
```

### Ensemble Strategy

Combines **global model** (Stage 1) and **regional models** (Stage 2) with optimized weights:

| Region | Global Weight | Regional Weight |
|--------|---------------|-----------------|
| 西北 (Northwest) | 40% | 60% |
| 华东 (East China) | 50% | 50% |
| 中南 (Central-South) | 65% | 35% |
| 东北 (Northeast) | 80% | 20% |
| 西南 (Southwest) | 80% | 20% |
| 华北 (North China) | 100% | 0% |


### Output Products

Four national-scale raster maps (GeoTIFF, 90m resolution):
```
maps_2020_ensemble_regional_adaptive/
├── pred_ensemble_adaptive_F_2020_90m.tif      # Farm-only
├── pred_ensemble_adaptive_F_NF_2020_90m.tif   # Farm + Non-farm
├── pred_ensemble_adaptive_NF_F_2020_90m.tif   # Non-farm + Farm
└── pred_ensemble_adaptive_NF_2020_90m.tif     # Non-farm-only
```

Each pixel contains probability values (0-1) for that livelihood strategy.

**Inference Time:** 8 hours (full China at 90m resolution)
**Auto-Checkpoint:** Saves progress every 1,000 batches - re-run to resume if interrupted.

---

## 🐛 Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch size in scripts
BATCH_SIZE = 128   # For training
BATCH_SIZE = 512   # For inference
```

### Slow Training
```python
USE_AMP = True          # Enable mixed precision
NUM_WORKERS = 4         # Increase workers (Linux/macOS only)
```

### Path Configuration

Update file paths in scripts to match your directory structure:
```python
DAY_TIFS = [
    r"YOUR_PATH\Landsat_RED_2020_90m_zscore.tif",
    # ... (7 bands total)
]
NIGHT_TIF = r"YOUR_PATH\VIIRS_2020_90m_zscore.tif"
LABEL_SHP = r"YOUR_PATH\Sample_2020.shp"
PROVINCE_SHP = r"YOUR_PATH\Provinces_China.shp"
```

### Windows Multiprocessing Issues
```python
NUM_WORKERS = 0  # Set to 0 if encountering errors on Windows
```

---

## 📂 Directory Structure
```
project/
├── Model/
│   ├── 00.Train_complete_Random-5-Fold-CV.py       # Stage 1 training
│   ├── 00.Train_Out-of-Region_5-fold-CV.py         # Stage 2 training
│   ├── 01.Plot_results.ipynb                        # Visualization
│   ├── 02.inference_2020_random_add_region_全像元.py # Inference
│   └── 说明.txt
│
├── Data/ (user-provided)
│   ├── Landsat_NL_Mector_90m_zscore/
│   │   ├── Landsat_RED_2020_90m_zscore.tif
│   │   ├── ... (7 Landsat bands)
│   │   └── VIIRS_2020_90m_zscore.tif
│   ├── sample_2020/
│   │   └── Sample_2020.shp
│   └── Province_boundary/
│       └── Provinces_China.shp
│
└── Outputs/
    ├── model_outputs_2020_resnet_optimized/         # Stage 1 outputs
    ├── model_outputs_2020_OUT_OF_REGION_MACRO_SOFT/ # Stage 2 outputs
    └── maps_2020_ensemble_regional_adaptive/        # Inference outputs
```

---

## 📚 Citation

If you use this code in your research, please cite:
```bibtex
@article{your_paper_2025,
  title={Deep Learning-Based Rural Livelihood Mapping Using Multispectral and Nighttime Light Imagery},
  author={Your Name et al.},
  journal={Journal Name},
  year={2025}
}
```

---

## 📧 Support

- **Issues:** [GitHub Issues](../../issues)
- **Documentation:** See individual script headers for detailed parameters
- **Contact:** your.email@example.com

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note:** Adjust all file paths in scripts according to your local directory structure before running.

## 📖 Citation

If you use this code in your research, please cite:

> Dawazhaxi, *et al*. "Mapping rural livelihood strategies to reveal the equality of urbanity." *Journal Name* **11**, 2583 (2026). https://doi.org/*****

**BibTeX:**
```bibtex
@article{dawazhaxi2026rural,
  title={Mapping rural livelihood strategies to reveal the socio-ecological impacts of rural development},
  author={Dawazhaxi and [Co-authors]},
  journal={Journal Name},
  volume={11},
  pages={2583},
  year={2026},
  doi={10.XXXX/XXXXX}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments
This work was supported by: *****
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
