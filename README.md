![plot](https://github.com/user-attachments/assets/122862bc-282c-4dd9-a4bb-47fb455fdccf)# Deep rural livelihoods model (DRLM) 🌾🏕💰🐏🐂🐖


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Paper](Currently writing)](https://*****)

> **Mapping rural livelihood strategies to reveal the equality of urbanity**
![Upload<svg width="1200" height="250" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <!-- 渐变背景 -->
    <linearGradient id="skyGradient" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#87CEEB;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#B0C4DE;stop-opacity:1" />
    </linearGradient>
    
    <!-- 地面渐变 -->
    <linearGradient id="groundGradient" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#228B22;stop-opacity:1" />
      <stop offset="20%" style="stop-color:#9ACD32;stop-opacity:1" />
      <stop offset="40%" style="stop-color:#DAA520;stop-opacity:1" />
      <stop offset="60%" style="stop-color:#D2B48C;stop-opacity:1" />
      <stop offset="80%" style="stop-color:#A9A9A9;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#696969;stop-opacity:1" />
    </linearGradient>
    
    <!-- 动画定义 -->
    <style>
      /* 云朵漂浮 */
      @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
      }
      
      /* 人物行走 */
      @keyframes walk {
        0% { transform: translateX(0px) scaleX(1); }
        25% { transform: translateX(30px) scaleX(1); }
        50% { transform: translateX(60px) scaleX(-1); }
        75% { transform: translateX(30px) scaleX(-1); }
        100% { transform: translateX(0px) scaleX(1); }
      }
      
      /* 拖拉机移动 */
      @keyframes tractor-move {
        0%, 100% { transform: translateX(0px); }
        50% { transform: translateX(40px); }
      }
      
      /* 火车移动 */
      @keyframes train-move {
        0% { transform: translateX(-100px); opacity: 0; }
        10% { opacity: 1; }
        90% { opacity: 1; }
        100% { transform: translateX(1300px); opacity: 0; }
      }
      
      /* 动物走动 */
      @keyframes animal-walk {
        0%, 100% { transform: translateX(0px); }
        50% { transform: translateX(-20px); }
      }
      
      /* 城市灯光闪烁 */
      @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
      }
      
      .cloud { animation: float 4s ease-in-out infinite; }
      .farmer { animation: walk 6s linear infinite; }
      .tractor { animation: tractor-move 8s ease-in-out infinite; }
      .train { animation: train-move 12s linear infinite; }
      .animal { animation: animal-walk 5s ease-in-out infinite; }
      .city-light { animation: blink 2s ease-in-out infinite; }
    </style>
  </defs>
  
  <!-- 天空背景 -->
  <rect width="1200" height="150" fill="url(#skyGradient)"/>
  
  <!-- 云朵 -->
  <g class="cloud">
    <ellipse cx="200" cy="30" rx="25" ry="15" fill="white" opacity="0.8"/>
    <ellipse cx="220" cy="30" rx="30" ry="18" fill="white" opacity="0.8"/>
  </g>
  <g class="cloud" style="animation-delay: 2s;">
    <ellipse cx="600" cy="50" rx="20" ry="12" fill="white" opacity="0.7"/>
    <ellipse cx="615" cy="50" rx="25" ry="15" fill="white" opacity="0.7"/>
  </g>
  <g class="cloud" style="animation-delay: 1s;">
    <ellipse cx="1000" cy="40" rx="30" ry="18" fill="white" opacity="0.8"/>
    <ellipse cx="1025" cy="40" rx="35" ry="20" fill="white" opacity="0.8"/>
  </g>
  
  <!-- 地面 -->
  <rect y="150" width="1200" height="100" fill="url(#groundGradient)"/>
  
  <!-- 区域1: 森林 (0-200) -->
  <g id="forest">
    <!-- 树木 -->
    <polygon points="50,150 40,180 60,180" fill="#2D5016"/>
    <rect x="47" y="180" width="6" height="20" fill="#654321"/>
    <polygon points="90,140 80,170 100,170" fill="#2D5016"/>
    <rect x="87" y="170" width="6" height="25" fill="#654321"/>
    <polygon points="130,145 120,175 140,175" fill="#2D5016"/>
    <rect x="127" y="175" width="6" height="22" fill="#654321"/>
    <polygon points="170,150 160,180 180,180" fill="#2D5016"/>
    <rect x="167" y="180" width="6" height="20" fill="#654321"/>
    
    <!-- 护林员 (行走) -->
    <g class="farmer" style="animation-delay: 0.5s;">
      <circle cx="80" cy="190" r="5" fill="#FDB462"/>
      <rect x="77" y="195" width="6" height="12" fill="#4A90E2"/>
      <line x1="80" y1="207" x2="75" y2="220" stroke="#654321" stroke-width="2"/>
      <line x1="80" y1="207" x2="85" y2="220" stroke="#654321" stroke-width="2"/>
    </g>
  </g>
  
  <!-- 区域2: 草原 (200-400) -->
  <g id="grassland">
    <!-- 草地纹理 -->
    <path d="M200,180 Q210,170 220,180 T240,180" stroke="#9ACD32" fill="none" stroke-width="2"/>
    <path d="M250,185 Q260,175 270,185 T290,185" stroke="#9ACD32" fill="none" stroke-width="2"/>
    <path d="M300,180 Q310,170 320,180 T340,180" stroke="#9ACD32" fill="none" stroke-width="2"/>
    
    <!-- 牛羊 (移动) -->
    <g class="animal">
      <ellipse cx="260" cy="200" rx="15" ry="10" fill="#8B7355"/>
      <circle cx="265" cy="195" r="6" fill="#8B7355"/>
      <line x1="253" y1="210" x2="253" y2="220" stroke="#654321" stroke-width="2"/>
      <line x1="267" y1="210" x2="267" y2="220" stroke="#654321" stroke-width="2"/>
    </g>
    
    <g class="animal" style="animation-delay: 1.5s;">
      <ellipse cx="320" cy="205" rx="12" ry="8" fill="#FFFFFF"/>
      <circle cx="324" cy="201" r="5" fill="#FFFFFF"/>
      <line x1="313" y1="213" x2="313" y2="220" stroke="#654321" stroke-width="1.5"/>
      <line x1="327" y1="213" x2="327" y2="220" stroke="#654321" stroke-width="1.5"/>
    </g>
    
    <!-- 牧民 -->
    <g class="farmer" style="animation-delay: 1s;">
      <circle cx="350" cy="190" r="5" fill="#FDB462"/>
      <rect x="347" y="195" width="6" height="12" fill="#E74C3C"/>
      <line x1="350" y1="207" x2="345" y2="220" stroke="#2C3E50" stroke-width="2"/>
      <line x1="350" y1="207" x2="355" y2="220" stroke="#2C3E50" stroke-width="2"/>
    </g>
  </g>
  
  <!-- 区域3: 农田 (400-600) -->
  <g id="farmland">
    <!-- 农田线条 -->
    <line x1="400" y1="190" x2="600" y2="190" stroke="#DAA520" stroke-width="2" stroke-dasharray="10,5"/>
    <line x1="400" y1="200" x2="600" y2="200" stroke="#DAA520" stroke-width="2" stroke-dasharray="10,5"/>
    <line x1="400" y1="210" x2="600" y2="210" stroke="#DAA520" stroke-width="2" stroke-dasharray="10,5"/>
    
    <!-- 拖拉机 (移动) -->
    <g class="tractor">
      <rect x="480" y="195" width="35" height="20" fill="#E74C3C"/>
      <circle cx="490" cy="220" r="8" fill="#2C3E50"/>
      <circle cx="510" cy="220" r="8" fill="#2C3E50"/>
      <rect x="495" y="185" width="10" height="10" fill="#3498DB"/>
      <!-- 烟囱 -->
      <rect x="500" y="180" width="3" height="5" fill="#95A5A6"/>
      <ellipse cx="501.5" cy="178" rx="4" ry="3" fill="#BDC3C7" opacity="0.6"/>
    </g>
    
    <!-- 农民 -->
    <g class="farmer" style="animation-delay: 2s;">
      <circle cx="550" cy="195" r="5" fill="#FDB462"/>
      <rect x="547" y="200" width="6" height="12" fill="#27AE60"/>
      <line x1="550" y1="212" x2="545" y2="225" stroke="#34495E" stroke-width="2"/>
      <line x1="550" y1="212" x2="555" y2="225" stroke="#34495E" stroke-width="2"/>
      <!-- 锄头 -->
      <line x1="558" y1="205" x2="568" y2="200" stroke="#8B4513" stroke-width="2"/>
    </g>
  </g>
  
  <!-- 区域4: 郊野/城郊 (600-800) -->
  <g id="suburban">
    <!-- 房屋 -->
    <rect x="650" y="175" width="30" height="35" fill="#E67E22"/>
    <polygon points="650,175 665,160 680,175" fill="#C0392B"/>
    <rect x="660" y="190" width="10" height="20" fill="#34495E"/>
    
    <rect x="720" y="170" width="35" height="40" fill="#F39C12"/>
    <polygon points="720,170 737.5,155 755,170" fill="#C0392B"/>
    <rect x="732" y="190" width="12" height="20" fill="#34495E"/>
    
    <!-- 道路 -->
    <rect x="600" y="215" width="200" height="8" fill="#7F8C8D"/>
    <line x1="600" y1="219" x2="800" y2="219" stroke="white" stroke-width="1" stroke-dasharray="15,10"/>
    
    <!-- 通勤人员 (骑车) -->
    <g class="farmer" style="animation-delay: 0.3s;">
      <circle cx="700" cy="208" r="4" fill="#FDB462"/>
      <line x1="700" y1="212" x2="700" y2="218" stroke="#3498DB" stroke-width="2"/>
      <circle cx="696" cy="220" r="3" fill="#2C3E50"/>
      <circle cx="704" cy="220" r="3" fill="#2C3E50"/>
      <line x1="696" y1="220" x2="704" y2="220" stroke="#7F8C8D" stroke-width="1.5"/>
    </g>
  </g>
  
  <!-- 区域5: 城镇 (800-1200) -->
  <g id="urban">
    <!-- 建筑群 -->
    <rect x="850" y="140" width="40" height="75" fill="#95A5A6"/>
    <rect x="865" y="150" width="10" height="8" fill="#F1C40F" class="city-light"/>
    <rect x="865" y="165" width="10" height="8" fill="#F1C40F" class="city-light" style="animation-delay: 0.5s;"/>
    <rect x="865" y="180" width="10" height="8" fill="#F1C40F" class="city-light" style="animation-delay: 1s;"/>
    
    <rect x="910" y="130" width="45" height="85" fill="#7F8C8D"/>
    <rect x="922" y="145" width="8" height="8" fill="#F39C12" class="city-light" style="animation-delay: 0.3s;"/>
    <rect x="935" y="145" width="8" height="8" fill="#F39C12" class="city-light" style="animation-delay: 0.8s;"/>
    <rect x="922" y="165" width="8" height="8" fill="#F39C12" class="city-light" style="animation-delay: 1.2s;"/>
    
    <rect x="975" y="145" width="38" height="70" fill="#BDC3C7"/>
    <rect x="987" y="160" width="8" height="8" fill="#E74C3C" class="city-light" style="animation-delay: 0.6s;"/>
    <rect x="987" y="178" width="8" height="8" fill="#E74C3C" class="city-light" style="animation-delay: 1.4s;"/>
    
    <rect x="1030" y="135" width="50" height="80" fill="#34495E"/>
    <rect x="1045" y="150" width="10" height="10" fill="#3498DB" class="city-light"/>
    <rect x="1060" y="150" width="10" height="10" fill="#3498DB" class="city-light" style="animation-delay: 0.7s;"/>
    <rect x="1045" y="170" width="10" height="10" fill="#3498DB" class="city-light" style="animation-delay: 1.1s;"/>
    
    <!-- 高铁轨道 -->
    <rect x="800" y="230" width="400" height="4" fill="#2C3E50"/>
    <rect x="800" y="232" width="10" height="2" fill="#95A5A6"/>
    <rect x="830" y="232" width="10" height="2" fill="#95A5A6"/>
    <rect x="860" y="232" width="10" height="2" fill="#95A5A6"/>
    <rect x="890" y="232" width="10" height="2" fill="#95A5A6"/>
    <rect x="920" y="232" width="10" height="2" fill="#95A5A6"/>
    <rect x="950" y="232" width="10" height="2" fill="#95A5A6"/>
    <rect x="980" y="232" width="10" height="2" fill="#95A5A6"/>
    <rect x="1010" y="232" width="10" height="2" fill="#95A5A6"/>
    <rect x="1040" y="232" width="10" height="2" fill="#95A5A6"/>
    <rect x="1070" y="232" width="10" height="2" fill="#95A5A6"/>
    
    <!-- 高铁 (快速移动) -->
    <g class="train">
      <rect x="900" y="218" width="80" height="14" fill="#E74C3C" rx="2"/>
      <rect x="905" y="221" width="12" height="8" fill="#3498DB"/>
      <rect x="925" y="221" width="12" height="8" fill="#3498DB"/>
      <rect x="945" y="221" width="12" height="8" fill="#3498DB"/>
      <rect x="965" y="221" width="12" height="8" fill="#3498DB"/>
      <polygon points="980,218 990,225 980,232" fill="#C0392B"/>
    </g>
    
    <!-- 上班族 -->
    <g class="farmer" style="animation-delay: 1.5s;">
      <circle cx="1100" cy="200" r="5" fill="#FDB462"/>
      <rect x="1097" y="205" width="6" height="12" fill="#2C3E50"/>
      <line x1="1100" y1="217" x2="1095" y2="230" stroke="#34495E" stroke-width="2"/>
      <line x1="1100" y1="217" x2="1105" y2="230" stroke="#34495E" stroke-width="2"/>
      <!-- 公文包 -->
      <rect x="1108" y="208" width="6" height="4" fill="#8B4513"/>
    </g>
  </g>
  
  <!-- 区域标签 -->
  <text x="100" y="245" font-family="Arial" font-size="12" fill="#2D5016" font-weight="bold">Forest</text>
  <text x="280" y="245" font-family="Arial" font-size="12" fill="#9ACD32" font-weight="bold">Grassland</text>
  <text x="480" y="245" font-family="Arial" font-size="12" fill="#DAA520" font-weight="bold">Farmland</text>
  <text x="680" y="245" font-family="Arial" font-size="12" fill="#E67E22" font-weight="bold">Suburban</text>
  <text x="1000" y="245" font-family="Arial" font-size="12" fill="#7F8C8D" font-weight="bold">Urban</text>
</svg>ing plot.svg…]()

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
