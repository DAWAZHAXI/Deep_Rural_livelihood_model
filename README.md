# Deep Rural Livelihood Model (DRLM)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18489933.svg)](https://doi.org/10.5281/zenodo.18489933)


**Mapping rural livelihood strategies in China using deep learning**

This repository provides the code for the **Deep Rural Livelihood Model (DRLM)**, developed for the Scientific Data manuscript *Mapping rural livelihood strategies in China using deep learning*. The model maps four rural livelihood strategy probabilities across rural China in 2020 using Landsat imagery, VIIRS nighttime lights, survey-derived livelihood labels, XGBoost-based sample expansion, and a dual-branch ResNet with Dirichlet regression.

The released dataset is available at Zenodo: [https://doi.org/10.5281/zenodo.18489933](https://doi.org/10.5281/zenodo.18489933).

![Framework of the Deep Rural Livelihood Model](readme_assets/image1.png)

## What You Can Do With This Repository

Use this repository to:

- prepare Landsat and VIIRS inputs for rural livelihood mapping;
- expand sparse survey-derived livelihood labels to rural settlement polygons;
- train the DRLM global model and region-informed models;
- run national-scale inference for four livelihood strategy probabilities;
- reproduce validation figures and diagnostic outputs used in the manuscript.

The repository is mainly intended for researchers who want to reproduce, inspect, or adapt the DRLM workflow. If you only need the final maps, download the GeoTIFF products directly from Zenodo.

## Model Inputs And Outputs

### Required Inputs

| Input | Role in the workflow | Notes |
| --- | --- | --- |
| Landsat 8/9 surface reflectance | Daytime multispectral input to DRLM | Red, green, blue, NIR, SWIR1, SWIR2, and thermal bands |
| VIIRS-DNB nighttime lights | Nighttime human-activity input to DRLM | Used as a separate model branch |
| Rural settlement polygons | Mapping and sample-expansion units | Derived from rural settlement data for 2020 |
| Survey-derived livelihood labels | Initial reference labels | Constructed from rural observation survey records |
| Township socioeconomic predictors | Sample expansion predictors | Used by quantile-regression XGBoost |
| Provincial boundaries | Region assignment | Used for macro-region model training and regional ensemble weighting |

### Model Outputs

DRLM predicts a four-part compositional vector at each valid rural pixel. Values range from 0 to 1 and sum to one.

| Output layer | Meaning |
| --- | --- |
| `F_2020_90m.tif` | Probability of farming-only livelihood strategy |
| `F_NF_2020_90m.tif` | Probability of farming-dominated mixed non-farming livelihood strategy |
| `NF_F_2020_90m.tif` | Probability of non-farming-dominated mixed farming livelihood strategy |
| `NF_2020_90m.tif` | Probability of non-farming-only livelihood strategy |
| `max_confidence_2020_90m.tif` | Maximum probability among the four components; useful for identifying dominant or ambiguous livelihood structure |

## Livelihood Strategy Classes

| Code | Livelihood strategy | Interpretation |
| --- | --- | --- |
| `F` | Farming-only | Households rely entirely or predominantly on agricultural production. |
| `F_NF` | Farming-dominated mixed non-farming | Farming is dominant, but non-farming activities provide supplementary income. |
| `NF_F` | Non-farming-dominated mixed farming | Non-farming income dominates, while farming remains a secondary activity. |
| `NF` | Non-farming-only | Households rely entirely or predominantly on non-agricultural employment or business. |

```text
F + F_NF + NF_F + NF = 1.0
```

## Quick Start

### 1. Clone The Repository

```bash
git clone https://github.com/DAWAZHAXI/Deep_Rural_livelihood_model.git
cd Deep_Rural_livelihood_model
```

### 2. Create The Environment

```bash
conda create -n rural_livelihood python=3.13
conda activate rural_livelihood
pip install -r requirements.txt
```

Check PyTorch and CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 3. Prepare Image And Sample Data

Run `Export_images_from_GEE.js` in the Google Earth Engine Code Editor to export the required Landsat and VIIRS layers. Then use `特征影像z_score处理.ipynb` to apply z-score normalization.

Expected image inputs:

```text
Landsat_NL_Mector_90m_zscore/
|-- Landsat_RED_2020_90m_zscore.tif
|-- Landsat_GREEN_2020_90m_zscore.tif
|-- Landsat_BLUE_2020_90m_zscore.tif
|-- Landsat_NIR_2020_90m_zscore.tif
|-- Landsat_SWIR1_2020_90m_zscore.tif
|-- Landsat_SWIR2_2020_90m_zscore.tif
|-- Landsat_TEMP1_2020_90m_zscore.tif
`-- VIIRS_2020_90m_zscore.tif
```

Prepare township and rural settlement samples using:

```text
全国乡镇代码匹配到乡镇调查和街景数据.ipynb
全国乡镇街道办事处牧场等重分类为0或1.ipynb
样本扩充_分位数XGBoost回归.ipynb
```

The sample-expansion notebook uses quantile-regression XGBoost to expand sparse survey-derived livelihood compositions to rural settlement polygons. The expanded layer is a **model-assisted label layer**, not direct household survey observations.

![Survey samples and sample-expansion diagnostics](readme_assets/image3.png)

### 4. Train The Global DRLM Model

```bash
python Model/00.Train_complete_Random-5-Fold-CV.py
```

This trains the global model with five-fold random cross-validation. Use `Model/01.Plot_results.ipynb` to summarize the cross-validation results.

### 5. Train Region-Informed Models

```bash
python Model/00.Train_Out-of-Region_5-fold-CV.py
```

This trains six macro-region models. These models are used together with the global model during national inference to better represent regional heterogeneity.

### 6. Run National Inference

```bash
python Model/02.inference_2020_random_add_region_全像元.py
```

The inference script combines global and region-informed predictions using fixed regional weights and generates four probability maps:

```text
maps_2020_ensemble_regional_adaptive/
|-- pred_ensemble_adaptive_F_2020_90m.tif
|-- pred_ensemble_adaptive_F_NF_2020_90m.tif
|-- pred_ensemble_adaptive_NF_F_2020_90m.tif
`-- pred_ensemble_adaptive_NF_2020_90m.tif
```

## Model Workflow

![Observed livelihood composition and sample expansion](readme_assets/image2.png)

The modelling workflow follows the manuscript:

1. Construct livelihood composition labels from rural observation survey records.
2. Geocode survey villages and aggregate household livelihood composition around observation sites.
3. Use rural settlement polygons as the mapping units.
4. Expand sparse survey labels to the national rural settlement layer using quantile-regression XGBoost.
5. Train DRLM using Landsat and VIIRS image patches.
6. Combine global and macro-region models for national inference.
7. Validate the product using internal reconstruction, region-informed holdout tests, and independent survey-based comparisons.

## DRLM Architecture

![DRLM architecture and transfer mapping workflow](readme_assets/image4.png)

DRLM uses two input branches:

- a daytime branch for seven Landsat bands;
- a nighttime branch for VIIRS-DNB nighttime lights.

The two branches are fused and passed through residual blocks. The prediction head outputs Dirichlet concentration parameters, which are normalized into four livelihood probabilities. This explicitly enforces the compositional constraint that the four components sum to one.

| Setting | Value |
| --- | --- |
| Daytime input | 7 Landsat bands |
| Nighttime input | 1 VIIRS-DNB band |
| Patch size | `64 x 64` pixels |
| Output shape | `[B, 4, 64, 64]` |
| Loss function | Dirichlet negative log-likelihood |
| Optimizer | AdamW with gradient clipping and learning-rate scheduling |

## Regional Ensemble Weights

The final national product combines the global model and the corresponding macro-region model using fixed weights.

| Macro-region | Global model weight | Region-informed model weight |
| --- | ---: | ---: |
| Northwest | 0.40 | 0.60 |
| East | 0.50 | 0.50 |
| Central South | 0.65 | 0.35 |
| Northeast | 0.80 | 0.20 |
| Southwest | 0.80 | 0.20 |
| North | 1.00 | 0.00 |

These weights are a transparent mapping rule based on regional model behavior, sample availability, and holdout performance. They are used to generate the released product and should not be interpreted as independent validation results.

## Example Output Maps

![Farming-only probability](readme_assets/image5.png)

*Farming-only (`F`) livelihood probability.*

![Farming-dominated mixed non-farming probability](readme_assets/image6.png)

*Farming-dominated mixed non-farming (`F_NF`) livelihood probability.*

![Non-farming-dominated mixed farming probability](readme_assets/image7.png)

*Non-farming-dominated mixed farming (`NF_F`) livelihood probability.*

![Non-farming-only probability](readme_assets/image8.png)

*Non-farming-only (`NF`) livelihood probability.*

## How To Interpret The Results

The model output should be interpreted as a probability composition, not as a hard class label. For example:

| Example probability vector | Suggested interpretation |
| --- | --- |
| `F=0.80, F_NF=0.10, NF_F=0.05, NF=0.05` | Strongly farming-oriented livelihood structure |
| `F=0.35, F_NF=0.30, NF_F=0.20, NF=0.15` | Mixed and relatively uncertain livelihood structure |
| `F=0.05, F_NF=0.10, NF_F=0.20, NF=0.65` | Strongly non-farming-oriented livelihood structure |

Use the full probability vector whenever possible. The dominant class is useful for visualization, but it discards important information about mixed livelihood strategies.

The `max_confidence_2020_90m.tif` layer can help identify areas where one livelihood component clearly dominates. Low confidence values indicate mixed or ambiguous livelihood composition and should be interpreted cautiously.

## Validation Summary

The manuscript uses a tiered validation framework because the released maps are spatial proxies derived from survey-informed modelling, not direct household observations.

![Cross-validation and baseline comparison](readme_assets/image9.png)

| Validation component | What it evaluates | Main interpretation |
| --- | --- | --- |
| Internal cross-validation | Agreement between DRLM predictions and expanded settlement-level labels | Useful for model reconstruction and stability, but not independent external validation |
| Region-informed validation | Spatial heterogeneity and regional transferability | Supports the need for regional weighting |
| Fixed rural observation survey validation | Agreement with independent survey-based livelihood composition | Stronger support for dominant categories; moderate support for mixed categories |
| CFPS county-level validation | Directional consistency with independent income-share indicators | Broad county-level check, not pixel-level validation |

Internal reconstruction against expanded settlement labels reached high R2 values:

| Component | R2 against expanded settlement labels |
| --- | ---: |
| `F` | 0.893 |
| `F_NF` | 0.868 |
| `NF_F` | 0.835 |
| `NF` | 0.823 |

Independent validation against 241 fixed rural observation sites showed the strongest agreement for the farming-only component (`R2 = 0.75`, `RMSE = 0.16`). Mixed livelihood categories showed moderate agreement, which is expected because transitional livelihood structures are more heterogeneous.

![Agreement with expanded settlement labels](readme_assets/image10.png)

![Maximum confidence diagnostics](readme_assets/image11.png)

![Independent fixed-site survey validation](readme_assets/image12.png)

![CFPS county-level validation](readme_assets/image13.png)

## Recommended Use Cases

This model and dataset are suitable for:

- regional analysis of rural livelihood structure;
- comparing farming-oriented and non-farming-oriented livelihood gradients;
- linking livelihood probabilities with land-use, food-security, ecosystem-service, biodiversity, carbon, or rural-development indicators;
- identifying areas for further field survey or regional case studies;
- supporting exploratory analysis of rural transformation under sustainability and rural revitalization contexts.

They are **not** suitable for:

- household-level livelihood inference;
- direct policy targeting of individual villages;
- program eligibility decisions;
- replacing contemporary household surveys;
- interpreting mixed livelihood areas as hard categorical classes.

## Hardware And Runtime Notes

| Component | Specification used or recommended |
| --- | --- |
| OS | Windows 10 or Linux |
| CPU | Intel Xeon W-2295 used in this study |
| RAM | 128 GB used; 64 GB minimum recommended |
| GPU | NVIDIA RTX A5000 24 GB used; 10 GB VRAM minimum recommended |
| Storage | 500 GB free space recommended for national-scale processing |

Approximate runtime reported in the project materials:

| Task | Approximate time |
| --- | ---: |
| Global and region-informed training | about 90 hours on GPU |
| Full national inference | about 8 hours on GPU |

## Troubleshooting

### Out Of Memory

Reduce batch size in the training or inference scripts:

```python
BATCH_SIZE = 128
```

### Windows Multiprocessing Issues

Set:

```python
NUM_WORKERS = 0
```

### Local Path Configuration

Update the paths inside the scripts before running:

```python
DAY_TIFS = [
    r"YOUR_PATH\Landsat_RED_2020_90m_zscore.tif",
    # other Landsat bands
]
NIGHT_TIF = r"YOUR_PATH\VIIRS_2020_90m_zscore.tif"
LABEL_SHP = r"YOUR_PATH\Sample_2020.shp"
PROVINCE_SHP = r"YOUR_PATH\Provinces_China.shp"
```

## Repository Structure

```text
project/
|-- Model/
|   |-- 00.Train_complete_Random-5-Fold-CV.py
|   |-- 00.Train_Out-of-Region_5-fold-CV.py
|   |-- 01.Plot_results.ipynb
|   |-- 02.inference_2020_random_add_region_全像元.py
|   `-- 说明.txt
|
|-- Data/ (user-provided)
|   |-- Landsat_NL_Mector_90m_zscore/
|   |-- sample_2020/
|   `-- Province_boundary/
|
|-- Outputs/
|   |-- model_outputs_2020_resnet_optimized/
|   |-- model_outputs_2020_OUT_OF_REGION_MACRO_SOFT/
|   `-- maps_2020_ensemble_regional_adaptive/
|
`-- readme_assets/
    |-- image1.png
    |-- ...
    `-- image13.png
```

## Data Availability

The rural livelihood probability maps, maximum-confidence layer, expanded settlement-level sample layer, model outputs, validation outputs, and supporting CSV files generated in this study are available from Zenodo:

[https://doi.org/10.5281/zenodo.18489933](https://doi.org/10.5281/zenodo.18489933)

Original household survey records, CFPS records, raw satellite imagery, and third-party source datasets are not redistributed.

## Citation

If you use this code or dataset, please cite the Zenodo record and the associated manuscript when available.

```bibtex
@dataset{dawa_2026_rural_livelihood,
  author    = {Dawa, Zhaxi and Yu, Wenjuan and Zhou, Weiqi},
  title     = {Mapping rural livelihood strategies in China using deep learning},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18489933},
  url       = {https://doi.org/10.5281/zenodo.18489933}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

This work was supported by the National Natural Science Foundation for Young Scientist Fund Program (No. 42501357), the Postdoctoral Fellowship Program (Grade B) of China Postdoctoral Science Foundation (No. GZB20250080), and the National Natural Science Fund for Distinguished Young Scholars (No. 42225104).

## Contact

**Lead author:** Dawa Zhaxi  
**GitHub:** [@DAWAZHAXI](https://github.com/DAWAZHAXI)  
**Email:** [15687851457@163.com](mailto:15687851457@163.com)
# Deep Rural Livelihood Model (DRLM)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18489933.svg)](https://doi.org/10.5281/zenodo.18489933)

**Mapping rural livelihood strategies in China using deep learning**

This repository provides the code for the **Deep Rural Livelihood Model (DRLM)**, developed for the Scientific Data manuscript *Mapping rural livelihood strategies in China using deep learning*. The model maps four rural livelihood strategy probabilities across rural China in 2020 using Landsat imagery, VIIRS nighttime lights, survey-derived livelihood labels, XGBoost-based sample expansion, and a dual-branch ResNet with Dirichlet regression.

The released dataset is available at Zenodo: [https://doi.org/10.5281/zenodo.18489933](https://doi.org/10.5281/zenodo.18489933).

![Framework of the Deep Rural Livelihood Model](readme_assets/image1.png)

## What You Can Do With This Repository

Use this repository to:

- prepare Landsat and VIIRS inputs for rural livelihood mapping;
- expand sparse survey-derived livelihood labels to rural settlement polygons;
- train the DRLM global model and region-informed models;
- run national-scale inference for four livelihood strategy probabilities;
- reproduce validation figures and diagnostic outputs used in the manuscript.

The repository is mainly intended for researchers who want to reproduce, inspect, or adapt the DRLM workflow. If you only need the final maps, download the GeoTIFF products directly from Zenodo.

## Model Inputs And Outputs

### Required Inputs

| Input | Role in the workflow | Notes |
| --- | --- | --- |
| Landsat 8/9 surface reflectance | Daytime multispectral input to DRLM | Red, green, blue, NIR, SWIR1, SWIR2, and thermal bands |
| VIIRS-DNB nighttime lights | Nighttime human-activity input to DRLM | Used as a separate model branch |
| Rural settlement polygons | Mapping and sample-expansion units | Derived from rural settlement data for 2020 |
| Survey-derived livelihood labels | Initial reference labels | Constructed from rural observation survey records |
| Township socioeconomic predictors | Sample expansion predictors | Used by quantile-regression XGBoost |
| Provincial boundaries | Region assignment | Used for macro-region model training and regional ensemble weighting |

### Model Outputs

DRLM predicts a four-part compositional vector at each valid rural pixel. Values range from 0 to 1 and sum to one.

| Output layer | Meaning |
| --- | --- |
| `F_2020_90m.tif` | Probability of farming-only livelihood strategy |
| `F_NF_2020_90m.tif` | Probability of farming-dominated mixed non-farming livelihood strategy |
| `NF_F_2020_90m.tif` | Probability of non-farming-dominated mixed farming livelihood strategy |
| `NF_2020_90m.tif` | Probability of non-farming-only livelihood strategy |
| `max_confidence_2020_90m.tif` | Maximum probability among the four components; useful for identifying dominant or ambiguous livelihood structure |

## Livelihood Strategy Classes

| Code | Livelihood strategy | Interpretation |
| --- | --- | --- |
| `F` | Farming-only | Households rely entirely or predominantly on agricultural production. |
| `F_NF` | Farming-dominated mixed non-farming | Farming is dominant, but non-farming activities provide supplementary income. |
| `NF_F` | Non-farming-dominated mixed farming | Non-farming income dominates, while farming remains a secondary activity. |
| `NF` | Non-farming-only | Households rely entirely or predominantly on non-agricultural employment or business. |

```text
F + F_NF + NF_F + NF = 1.0
```

## Quick Start

### 1. Clone The Repository

```bash
git clone https://github.com/DAWAZHAXI/Deep_Rural_livelihood_model.git
cd Deep_Rural_livelihood_model
```

### 2. Create The Environment

```bash
conda create -n rural_livelihood python=3.13
conda activate rural_livelihood
pip install -r requirements.txt
```

Check PyTorch and CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 3. Prepare Image And Sample Data

Run `Export_images_from_GEE.js` in the Google Earth Engine Code Editor to export the required Landsat and VIIRS layers. Then use `特征影像z_score处理.ipynb` to apply z-score normalization.

Expected image inputs:

```text
Landsat_NL_Mector_90m_zscore/
|-- Landsat_RED_2020_90m_zscore.tif
|-- Landsat_GREEN_2020_90m_zscore.tif
|-- Landsat_BLUE_2020_90m_zscore.tif
|-- Landsat_NIR_2020_90m_zscore.tif
|-- Landsat_SWIR1_2020_90m_zscore.tif
|-- Landsat_SWIR2_2020_90m_zscore.tif
|-- Landsat_TEMP1_2020_90m_zscore.tif
`-- VIIRS_2020_90m_zscore.tif
```

Prepare township and rural settlement samples using:

```text
全国乡镇代码匹配到乡镇调查和街景数据.ipynb
全国乡镇街道办事处牧场等重分类为0或1.ipynb
样本扩充_分位数XGBoost回归.ipynb
```

The sample-expansion notebook uses quantile-regression XGBoost to expand sparse survey-derived livelihood compositions to rural settlement polygons. The expanded layer is a **model-assisted label layer**, not direct household survey observations.

![Survey samples and sample-expansion diagnostics](readme_assets/image3.png)

### 4. Train The Global DRLM Model

```bash
python Model/00.Train_complete_Random-5-Fold-CV.py
```

This trains the global model with five-fold random cross-validation. Use `Model/01.Plot_results.ipynb` to summarize the cross-validation results.

### 5. Train Region-Informed Models

```bash
python Model/00.Train_Out-of-Region_5-fold-CV.py
```

This trains six macro-region models. These models are used together with the global model during national inference to better represent regional heterogeneity.

### 6. Run National Inference

```bash
python Model/02.inference_2020_random_add_region_全像元.py
```

The inference script combines global and region-informed predictions using fixed regional weights and generates four probability maps:

```text
maps_2020_ensemble_regional_adaptive/
|-- pred_ensemble_adaptive_F_2020_90m.tif
|-- pred_ensemble_adaptive_F_NF_2020_90m.tif
|-- pred_ensemble_adaptive_NF_F_2020_90m.tif
`-- pred_ensemble_adaptive_NF_2020_90m.tif
```

## Model Workflow

![Observed livelihood composition and sample expansion](readme_assets/image2.png)

The modelling workflow follows the manuscript:

1. Construct livelihood composition labels from rural observation survey records.
2. Geocode survey villages and aggregate household livelihood composition around observation sites.
3. Use rural settlement polygons as the mapping units.
4. Expand sparse survey labels to the national rural settlement layer using quantile-regression XGBoost.
5. Train DRLM using Landsat and VIIRS image patches.
6. Combine global and macro-region models for national inference.
7. Validate the product using internal reconstruction, region-informed holdout tests, and independent survey-based comparisons.

## DRLM Architecture

![DRLM architecture and transfer mapping workflow](readme_assets/image4.png)

DRLM uses two input branches:

- a daytime branch for seven Landsat bands;
- a nighttime branch for VIIRS-DNB nighttime lights.

The two branches are fused and passed through residual blocks. The prediction head outputs Dirichlet concentration parameters, which are normalized into four livelihood probabilities. This explicitly enforces the compositional constraint that the four components sum to one.

| Setting | Value |
| --- | --- |
| Daytime input | 7 Landsat bands |
| Nighttime input | 1 VIIRS-DNB band |
| Patch size | `64 x 64` pixels |
| Output shape | `[B, 4, 64, 64]` |
| Loss function | Dirichlet negative log-likelihood |
| Optimizer | AdamW with gradient clipping and learning-rate scheduling |

## Regional Ensemble Weights

The final national product combines the global model and the corresponding macro-region model using fixed weights.

| Macro-region | Global model weight | Region-informed model weight |
| --- | ---: | ---: |
| Northwest | 0.40 | 0.60 |
| East | 0.50 | 0.50 |
| Central South | 0.65 | 0.35 |
| Northeast | 0.80 | 0.20 |
| Southwest | 0.80 | 0.20 |
| North | 1.00 | 0.00 |

These weights are a transparent mapping rule based on regional model behavior, sample availability, and holdout performance. They are used to generate the released product and should not be interpreted as independent validation results.

## Example Output Maps

![Farming-only probability](readme_assets/image5.png)

*Farming-only (`F`) livelihood probability.*

![Farming-dominated mixed non-farming probability](readme_assets/image6.png)

*Farming-dominated mixed non-farming (`F_NF`) livelihood probability.*

![Non-farming-dominated mixed farming probability](readme_assets/image7.png)

*Non-farming-dominated mixed farming (`NF_F`) livelihood probability.*

![Non-farming-only probability](readme_assets/image8.png)

*Non-farming-only (`NF`) livelihood probability.*

## How To Interpret The Results

The model output should be interpreted as a probability composition, not as a hard class label. For example:

| Example probability vector | Suggested interpretation |
| --- | --- |
| `F=0.80, F_NF=0.10, NF_F=0.05, NF=0.05` | Strongly farming-oriented livelihood structure |
| `F=0.35, F_NF=0.30, NF_F=0.20, NF=0.15` | Mixed and relatively uncertain livelihood structure |
| `F=0.05, F_NF=0.10, NF_F=0.20, NF=0.65` | Strongly non-farming-oriented livelihood structure |

Use the full probability vector whenever possible. The dominant class is useful for visualization, but it discards important information about mixed livelihood strategies.

The `max_confidence_2020_90m.tif` layer can help identify areas where one livelihood component clearly dominates. Low confidence values indicate mixed or ambiguous livelihood composition and should be interpreted cautiously.

## Validation Summary

The manuscript uses a tiered validation framework because the released maps are spatial proxies derived from survey-informed modelling, not direct household observations.

![Cross-validation and baseline comparison](readme_assets/image9.png)

| Validation component | What it evaluates | Main interpretation |
| --- | --- | --- |
| Internal cross-validation | Agreement between DRLM predictions and expanded settlement-level labels | Useful for model reconstruction and stability, but not independent external validation |
| Region-informed validation | Spatial heterogeneity and regional transferability | Supports the need for regional weighting |
| Fixed rural observation survey validation | Agreement with independent survey-based livelihood composition | Stronger support for dominant categories; moderate support for mixed categories |
| CFPS county-level validation | Directional consistency with independent income-share indicators | Broad county-level check, not pixel-level validation |

Internal reconstruction against expanded settlement labels reached high R2 values:

| Component | R2 against expanded settlement labels |
| --- | ---: |
| `F` | 0.893 |
| `F_NF` | 0.868 |
| `NF_F` | 0.835 |
| `NF` | 0.823 |

Independent validation against 241 fixed rural observation sites showed the strongest agreement for the farming-only component (`R2 = 0.75`, `RMSE = 0.16`). Mixed livelihood categories showed moderate agreement, which is expected because transitional livelihood structures are more heterogeneous.

![Agreement with expanded settlement labels](readme_assets/image10.png)

![Maximum confidence diagnostics](readme_assets/image11.png)

![Independent fixed-site survey validation](readme_assets/image12.png)

![CFPS county-level validation](readme_assets/image13.png)

## Recommended Use Cases

This model and dataset are suitable for:

- regional analysis of rural livelihood structure;
- comparing farming-oriented and non-farming-oriented livelihood gradients;
- linking livelihood probabilities with land-use, food-security, ecosystem-service, biodiversity, carbon, or rural-development indicators;
- identifying areas for further field survey or regional case studies;
- supporting exploratory analysis of rural transformation under sustainability and rural revitalization contexts.

They are **not** suitable for:

- household-level livelihood inference;
- direct policy targeting of individual villages;
- program eligibility decisions;
- replacing contemporary household surveys;
- interpreting mixed livelihood areas as hard categorical classes.

## Hardware And Runtime Notes

| Component | Specification used or recommended |
| --- | --- |
| OS | Windows 10 or Linux |
| CPU | Intel Xeon W-2295 used in this study |
| RAM | 128 GB used; 64 GB minimum recommended |
| GPU | NVIDIA RTX A5000 24 GB used; 10 GB VRAM minimum recommended |
| Storage | 500 GB free space recommended for national-scale processing |

Approximate runtime reported in the project materials:

| Task | Approximate time |
| --- | ---: |
| Global and region-informed training | about 90 hours on GPU |
| Full national inference | about 8 hours on GPU |

## Troubleshooting

### Out Of Memory

Reduce batch size in the training or inference scripts:

```python
BATCH_SIZE = 128
```

### Windows Multiprocessing Issues

Set:

```python
NUM_WORKERS = 0
```

### Local Path Configuration

Update the paths inside the scripts before running:

```python
DAY_TIFS = [
    r"YOUR_PATH\Landsat_RED_2020_90m_zscore.tif",
    # other Landsat bands
]
NIGHT_TIF = r"YOUR_PATH\VIIRS_2020_90m_zscore.tif"
LABEL_SHP = r"YOUR_PATH\Sample_2020.shp"
PROVINCE_SHP = r"YOUR_PATH\Provinces_China.shp"
```

## Repository Structure

```text
project/
|-- Model/
|   |-- 00.Train_complete_Random-5-Fold-CV.py
|   |-- 00.Train_Out-of-Region_5-fold-CV.py
|   |-- 01.Plot_results.ipynb
|   |-- 02.inference_2020_random_add_region_全像元.py
|   `-- 说明.txt
|
|-- Data/ (user-provided)
|   |-- Landsat_NL_Mector_90m_zscore/
|   |-- sample_2020/
|   `-- Province_boundary/
|
|-- Outputs/
|   |-- model_outputs_2020_resnet_optimized/
|   |-- model_outputs_2020_OUT_OF_REGION_MACRO_SOFT/
|   `-- maps_2020_ensemble_regional_adaptive/
|
`-- readme_assets/
    |-- image1.png
    |-- ...
    `-- image13.png
```

## Data Availability

The rural livelihood probability maps, maximum-confidence layer, expanded settlement-level sample layer, model outputs, validation outputs, and supporting CSV files generated in this study are available from Zenodo:

[https://doi.org/10.5281/zenodo.18489933](https://doi.org/10.5281/zenodo.18489933)

Original household survey records, CFPS records, raw satellite imagery, and third-party source datasets are not redistributed.

## Citation

If you use this code or dataset, please cite the Zenodo record and the associated manuscript when available.

```bibtex
@dataset{dawa_2026_rural_livelihood,
  author    = {Dawa, Zhaxi and Yu, Wenjuan and Zhou, Weiqi},
  title     = {Mapping rural livelihood strategies in China using deep learning},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18489933},
  url       = {https://doi.org/10.5281/zenodo.18489933}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

This work was supported by the National Natural Science Foundation for Young Scientist Fund Program (No. 42501357), the Postdoctoral Fellowship Program (Grade B) of China Postdoctoral Science Foundation (No. GZB20250080), and the National Natural Science Fund for Distinguished Young Scholars (No. 42225104).

## Contact

**Lead author:** Dawa Zhaxi  
**GitHub:** [@DAWAZHAXI](https://github.com/DAWAZHAXI)  
**Email:** [15687851457@163.com](mailto:15687851457@163.com)
