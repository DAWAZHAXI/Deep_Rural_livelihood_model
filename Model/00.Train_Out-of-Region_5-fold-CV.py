#!/usr/bin/env python
# coding: utf-8
"""
Out-of-Region 宏区验证训练脚本（宽松版 + 断点续跑）
================================================
在原 Out-of-Province 脚本基础上改为“宏观区域”划分，并实现：

方案 2：宽松版 Out-of-Region（目标宏区部分样本参与训练）
-------------------------------------------------------
设定 6 个宏区：东北、华北、华东、中南、西南、西北。
对每个宏区 R：

1）将该宏区所有样本集合记为 region_idx_all。
2）其中一部分样本作为真正测试集 test_idx（不参与训练也不参与验证）；
3）剩余样本 pool_target 与其它宏区所有样本 other_idx_all 一起组成
   train+val 池，在其中随机划分 train / val。

这样：
- 测试集仍然是“目标宏区内部没见过的点”；
- 但模型在训练时已经“见过这个宏区的大部分统计结构”，
  是一个更宽松、更现实的 Out-of-Region 设定。
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.metrics import r2_score
import rasterio
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time

print("="*80)
print("🌍 Out-of-Region 宏区验证训练（宽松版 + 断点续跑）")
print("="*80)

# ============================================================================
#                           第一部分：配置参数
# ============================================================================

# 路径配置
DAY_TIFS = [
    r"F:\Landsat_NL_Mector_90m_zscore\Landsat_RED_2020_90m_zscore.tif",
    r"F:\Landsat_NL_Mector_90m_zscore\Landsat_GREEN_2020_90m_zscore.tif",
    r"F:\Landsat_NL_Mector_90m_zscore\Landsat_BLUE_2020_90m_zscore.tif",
    r"F:\Landsat_NL_Mector_90m_zscore\Landsat_NIR_2020_90m_zscore.tif",
    r"F:\Landsat_NL_Mector_90m_zscore\Landsat_SWIR1_2020_90m_zscore.tif",
    r"F:\Landsat_NL_Mector_90m_zscore\Landsat_SWIR2_2020_90m_zscore.tif",
    r"F:\Landsat_NL_Mector_90m_zscore\Landsat_TEMP1_2020_90m_zscore.tif",
]
NIGHT_TIF = r"F:\Landsat_NL_Mector_90m_zscore\VIIRS_2020_90m_zscore.tif"
LABEL_SHP = r"F:\sample_2020\Sample_2020.shp"
PROVINCE_SHP = r"F:\Province_boundary\Provinces_China.shp"
TARGET_FIELDS = ["F", "F_NF", "NF_F", "NF"]

OUT_DIR = r"F:\model_outputs_2020_OUT_OF_REGION_MACRO_SOFT"
os.makedirs(OUT_DIR, exist_ok=True)

# ⭐ 检查点目录
CHECKPOINT_DIR = os.path.join(OUT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 训练参数
BEST_LR = 3e-4
BEST_WD = 1e-3
PATCH_SIZE = 64
BATCH_SIZE = 256
EPOCHS = 300
PATIENCE = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0

# ⭐ 检查点保存频率
CHECKPOINT_INTERVAL = 10  # 每10个epoch保存一次

# ⭐ 宏区宽松划分参数（你可以按需要改）
TARGET_TEST_RATIO = 0.2   # 目标宏区样本中用于 test 的比例（例如 0.2 = 20% 留作真正测试）
GLOBAL_VAL_RATIO = 0.2    # train+val 池中用于验证的比例（例如 0.2 = 80% 训练，20% 验证）

print(f"✅ 配置完成")
print(f"   设备: {DEVICE}")
print(f"   输出目录: {OUT_DIR}")
print(f"   检查点目录: {CHECKPOINT_DIR}")
print(f"   检查点间隔: 每{CHECKPOINT_INTERVAL}个epoch")
print(f"   宏区 test 比例: {TARGET_TEST_RATIO:.2f}, train/val 池 val 比例: {GLOBAL_VAL_RATIO:.2f}")

# ============================================================================
#                    第二部分：数据集类（保持原有功能）
# ============================================================================

class PatchPointDataset(Dataset):
    """点样本数据集（内存缓存版）"""
    
    def __init__(self, shp_path, day_paths, night_path, target_fields,
                 patch_size=64, mode='train', check_valid=True, 
                 max_samples=None, cache_data=True):
        super().__init__()
        
        print(f"📂 读取shapefile: {os.path.basename(shp_path)}")
        self.gdf = gpd.read_file(shp_path).reset_index(drop=True)
        
        if max_samples is not None and max_samples < len(self.gdf):
            print(f"⚠️ 快速测试模式：只使用前 {max_samples} 个样本")
            self.gdf = self.gdf.iloc[:max_samples]
        
        self.day_paths = day_paths
        self.night_path = night_path
        self.target_fields = target_fields
        self.patch = patch_size
        self.mode = mode
        self.cache_data = cache_data
        
        print(f"🔧 打开栅格文件...")
        self.day_srcs = [rasterio.open(p) for p in self.day_paths]
        self.night_src = rasterio.open(self.night_path)
        
        self.height = self.day_srcs[0].height
        self.width = self.day_srcs[0].width
        self.transform = self.day_srcs[0].transform
        
        print(f"   影像尺寸: {self.height} × {self.width}")
        
        if check_valid:
            self.valid_idx, self.cached_patches = self._build_valid_index_and_cache()
        else:
            self.valid_idx = list(range(len(self.gdf)))
            self.cached_patches = None
            print(f"✅ 跳过预筛，使用所有 {len(self.gdf)} 个样本")
        
        if len(self.valid_idx) == 0:
            raise RuntimeError("没有可用样本")
        
        if self.cache_data and self.cached_patches is not None:
            print("💾 数据已缓存到内存，关闭栅格文件")
            for s in self.day_srcs:
                s.close()
            self.night_src.close()
            self.day_srcs = None
            self.night_src = None
    
    def _check_and_load_sample(self, idx):
        """检查样本有效性并加载数据到内存"""
        try:
            row = self.gdf.iloc[idx]
            geom = row.geometry
            if geom is None or geom.is_empty:
                return None
            if geom.geom_type != "Point":
                geom = geom.centroid
            x, y = geom.x, geom.y
            
            half = self.patch // 2
            first_src = self.day_srcs[0]
            r, c = rasterio.transform.rowcol(first_src.transform, x, y)
            
            if (r < half or c < half or 
                r >= first_src.height - half or 
                c >= first_src.width - half):
                return None
            
            window = rasterio.windows.Window(c - half, r - half, self.patch, self.patch)
            
            day_stack = []
            for src in self.day_srcs:
                arr = src.read(1, window=window, boundless=True, masked=True)
                if arr.mask.all() or np.isnan(arr.filled(0)).all():
                    return None
                day_stack.append(arr.filled(0).astype(np.float32))
            
            day_arr = np.stack(day_stack, axis=0)
            
            night_arr = self.night_src.read(1, window=window, boundless=True, masked=True)
            if night_arr.mask.all() or np.isnan(night_arr.filled(0)).all():
                return None
            night_arr = night_arr.filled(0).astype(np.float32)[np.newaxis, :, :]
            
            vals = [row[f] for f in self.target_fields]
            if any((v is None) or (isinstance(v, float) and np.isnan(v)) for v in vals):
                return None
            
            y = np.array(vals, dtype=np.float32)
            s = y.sum()
            if s > 1e-6:
                y = y / s
            else:
                y = np.array([1.0] + [0.0] * (len(self.target_fields) - 1), dtype=np.float32)
            
            return {
                'day': day_arr,
                'night': night_arr,
                'y': y
            }
        
        except Exception:
            return None
    
    def _build_valid_index_and_cache(self):
        """多线程预筛并缓存数据到内存"""
        print("⏳ 多线程预筛并缓存数据到内存...")
        
        valid_idx = []
        cached_data = {} if self.cache_data else None
        
        max_workers = min(8, os.cpu_count() or 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._check_and_load_sample, idx): idx 
                      for idx in range(len(self.gdf))}
            
            for future in tqdm(as_completed(futures), 
                             total=len(futures), 
                             desc="加载数据"):
                idx = futures[future]
                result = future.result()
                
                if result is not None:
                    valid_idx.append(idx)
                    if self.cache_data:
                        cached_data[idx] = result
        
        valid_idx.sort()
        
        if self.cache_data and cached_data:
            sample_size = (cached_data[valid_idx[0]]['day'].nbytes + 
                          cached_data[valid_idx[0]]['night'].nbytes +
                          cached_data[valid_idx[0]]['y'].nbytes) / 1e6
            total_size = sample_size * len(valid_idx)
            print(f"✅ 有效样本: {len(valid_idx)}/{len(self.gdf)} ({len(valid_idx)/len(self.gdf)*100:.1f}%)")
            print(f"   内存占用: {total_size:.1f} MB ({sample_size:.2f} MB/样本)")
        else:
            print(f"✅ 有效样本: {len(valid_idx)}/{len(self.gdf)} ({len(valid_idx)/len(self.gdf)*100:.1f}%)")
        
        return valid_idx, cached_data
    
    def __len__(self):
        return len(self.valid_idx)
    
    def set_mode(self, mode: str):
        """切换模式"""
        assert mode in ["train", "val", "test"]
        self.mode = mode
    
    def _augment_day(self, day):
        """数据增强"""
        if np.random.rand() < 0.5:
            delta = np.random.uniform(-0.5, 0.5)
            day = day + delta
        if np.random.rand() < 0.5:
            factor = np.random.uniform(0.75, 1.25)
            day = day * factor
        return day
    
    def __getitem__(self, idx):
        real_idx = self.valid_idx[idx]
        
        if self.cached_patches is not None:
            data = self.cached_patches[real_idx]
            day = data['day'].copy()
            night = data['night'].copy()
            y = data['y'].copy()
        else:
            row = self.gdf.iloc[real_idx]
            geom = row.geometry
            if geom.geom_type != "Point":
                geom = geom.centroid
            x, y_coord = geom.x, geom.y
            r, c = rasterio.transform.rowcol(self.transform, x, y_coord)
            
            half = self.patch // 2
            window = rasterio.windows.Window(c - half, r - half, self.patch, self.patch)
            
            day_stack = []
            for src in self.day_srcs:
                arr = src.read(1, window=window, boundless=True, masked=True)
                day_stack.append(arr.filled(0).astype(np.float32))
            day = np.stack(day_stack, axis=0)
            
            night_arr = self.night_src.read(1, window=window, boundless=True, masked=True)
            night = night_arr.filled(0).astype(np.float32)[np.newaxis, :, :]
            
            vals = []
            for f in self.target_fields:
                v = row[f]
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    v = 0.0
                vals.append(float(v))
            y = np.array(vals, dtype=np.float32)
            s = y.sum()
            if s > 1e-6:
                y = y / s
            else:
                y = np.array([1.0] + [0.0] * (len(self.target_fields) - 1), dtype=np.float32)
        
        if self.mode == 'train':
            if np.random.rand() < 0.5:
                day = np.flip(day, axis=1).copy()
                night = np.flip(night, axis=1).copy()
            if np.random.rand() < 0.5:
                day = np.flip(day, axis=2).copy()
                night = np.flip(night, axis=2).copy()
            k = np.random.randint(0, 4)
            if k > 0:
                day = np.rot90(day, k, axes=(1, 2)).copy()
                night = np.rot90(night, k, axes=(1, 2)).copy()
            day = self._augment_day(day)
        
        return {
            "day": torch.from_numpy(day),
            "night": torch.from_numpy(night),
            "y": torch.from_numpy(y),
        }
    
    def close(self):
        """关闭文件句柄"""
        if self.day_srcs is not None:
            for s in self.day_srcs:
                if not s.closed:
                    s.close()
        if self.night_src is not None and not self.night_src.closed:
            self.night_src.close()

# ============================================================================
#                    第三部分：模型定义（保持原来结构）
# ============================================================================

class PreActBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 
                                     kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = None
    
    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        return out + shortcut


class ResNetStem(nn.Module):
    def __init__(self, in_channels, base_channels=64, num_blocks=5):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, 
                                stride=1, padding=1, bias=False)
        self.blocks = nn.Sequential(
            *[PreActBlock(base_channels, base_channels, stride=1) 
              for _ in range(num_blocks)]
        )
        self.bn_out = nn.BatchNorm2d(base_channels)
        self.relu_out = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.bn_out(x)
        x = self.relu_out(x)
        return x


class FusionResNetDirichlet(nn.Module):
    def __init__(self, day_channels=7, night_channels=1, n_comp=4,
                 base_channels=64, day_blocks=3, night_blocks=3, shared_blocks=5):
        super().__init__()
        self.day_stem = ResNetStem(day_channels, base_channels, day_blocks)
        self.night_stem = ResNetStem(night_channels, base_channels, night_blocks)
        
        self.shared_in_conv = nn.Conv2d(2 * base_channels, base_channels, 
                                       kernel_size=3, stride=1, padding=1, bias=False)
        self.shared_blocks = nn.Sequential(
            *[PreActBlock(base_channels, base_channels, stride=1) 
              for _ in range(shared_blocks)]
        )
        self.shared_bn = nn.BatchNorm2d(base_channels)
        self.shared_relu = nn.ReLU(inplace=True)
        
        self.head_conv = nn.Conv2d(base_channels, base_channels, 
                                   kernel_size=3, stride=1, padding=1, bias=False)
        self.head_bn = nn.BatchNorm2d(base_channels)
        self.head_relu = nn.ReLU(inplace=True)
        self.head_drop = nn.Dropout2d(0.1)
        self.head_out = nn.Conv2d(base_channels, n_comp, kernel_size=1, bias=True)
    
    def forward(self, day, night):
        d = self.day_stem(day)
        n = self.night_stem(night)
        x = torch.cat([d, n], dim=1)
        x = self.shared_in_conv(x)
        x = self.shared_blocks(x)
        x = self.shared_bn(x)
        x = self.shared_relu(x)
        x = self.head_conv(x)
        x = self.head_bn(x)
        x = self.head_relu(x)
        x = self.head_drop(x)
        alpha_raw = self.head_out(x)
        alpha = F.softplus(alpha_raw) + 1.0
        return alpha


def dirichlet_nll(y_true, alpha, eps=1e-7):
    """Dirichlet负对数似然损失"""
    if y_true.dim() == 2:
        pass
    elif y_true.dim() > 2:
        B, C, H, W = y_true.shape
        yc = y_true[:, :, H // 2, W // 2]
        y_true = yc
    
    B, C, H, W = alpha.shape
    ac = alpha[:, :, H // 2, W // 2]
    
    alpha0 = ac.sum(dim=1, keepdim=True)
    logC = torch.lgamma(alpha0) - torch.lgamma(ac).sum(dim=1, keepdim=True)
    
    y_safe = torch.clamp(y_true, min=eps, max=1.0 - eps)
    logL = logC + ((ac - 1.0) * torch.log(y_safe)).sum(dim=1, keepdim=True)
    return -logL.mean()


def r2_score_numpy(y_true, y_pred):
    """计算R²分数"""
    mask = np.isfinite(y_true).all(axis=1) & np.isfinite(y_pred).all(axis=1)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.shape[0] < 2:
        return np.nan, [np.nan] * y_true.shape[1]
    
    r2_each = [r2_score(y_true[:, i], y_pred[:, i]) 
               for i in range(y_true.shape[1])]
    r2_mean = float(np.mean(r2_each))
    return r2_mean, r2_each

# ============================================================================
#            第四部分：按“宏观区域”划分数据（宽松版 Out-of-Region）
# ============================================================================

# 宏区定义（可以按你自己习惯调整，但要和省名一致）
MACRO_REGION_DEF = {
    "东北": ["辽宁省", "吉林省", "黑龙江省"],
    "华北": ["北京市", "天津市", "河北省", "山西省", "内蒙古自治区"],
    "华东": ["上海市", "江苏省", "浙江省", "安徽省", "福建省", "江西省", "山东省"],
    "中南": ["河南省", "湖北省", "湖南省", "广东省", "广西壮族自治区", "海南省"],
    "西南": ["重庆市", "四川省", "贵州省", "云南省"],
    "西北": ["陕西省", "甘肃省", "青海省", "宁夏回族自治区", "新疆维吾尔自治区"],
}


def create_macroregion_folds_soft(base_ds, province_shp,
                                  target_test_ratio=0.2,
                                  global_val_ratio=0.2):
    """
    宏观区域划分（宽松版 Out-of-Region）

    对每个宏区 R：
    - region_idx_all: 该宏区所有有效样本
    - 其中 target_test_ratio 部分作为 test_idx（真正测试集）
    - 剩余 pool_target 与其它宏区样本 other_idx_all 合并成 train+val 池，
      再按 global_val_ratio 划分出 val_idx，剩余为 train_idx

    返回:
        folds: 长度 = 宏区数，每个元素是 dict:
            {
                "macro_region": str,
                "train_provinces": [...],
                "test_provinces": [...],
                "train_idx": np.array,
                "val_idx": np.array,
                "test_idx": np.array,
            }
        province_samples: {省名: [dataset_idx, ...]}
    """
    print("\n" + "="*80)
    print("🗺️  按宏观区域划分数据集（宽松版 Out-of-Region）")
    print("="*80)
    
    print("\n📂 读取省界数据...")
    provinces_gdf = gpd.read_file(province_shp)
    
    prov_field = '省'
    if prov_field not in provinces_gdf.columns:
        raise ValueError(f"省界shapefile中没有'{prov_field}'字段")
    
    # 排除港澳台
    exclude_regions = ['香港特别行政区', '澳门特别行政区', '台湾省']
    provinces_gdf = provinces_gdf[~provinces_gdf[prov_field].isin(exclude_regions)]
    
    print(f"   有效省份: {len(provinces_gdf)}")
    
    print("\n🔗 为有效样本分配省份...")
    valid_gdf = base_ds.gdf.iloc[base_ds.valid_idx].copy()
    valid_gdf['dataset_idx'] = range(len(base_ds.valid_idx))
    
    print(f"   有效样本数: {len(valid_gdf)}")
    
    # CRS 对齐
    if valid_gdf.crs != provinces_gdf.crs:
        print("   ⚠️ CRS不一致，重投影中...")
        valid_gdf = valid_gdf.to_crs(provinces_gdf.crs)
    
    samples_with_prov = gpd.sjoin(
        valid_gdf,
        provinces_gdf[[prov_field, 'geometry']],
        how='left',
        predicate='within'
    )
    
    n_unassigned = samples_with_prov[prov_field].isna().sum()
    if n_unassigned > 0:
        print(f"   ⚠️ {n_unassigned} 个样本未分配到省份（边界点），将被移除")
        samples_with_prov = samples_with_prov[samples_with_prov[prov_field].notna()]
    
    # 统计每个省份的样本 index
    print("\n📊 样本在各省份分布:")
    province_samples = {}
    for prov_name in sorted(samples_with_prov[prov_field].unique()):
        prov_mask = samples_with_prov[prov_field] == prov_name
        prov_indices = samples_with_prov[prov_mask]['dataset_idx'].tolist()
        province_samples[prov_name] = prov_indices
        print(f"   {prov_name}: {len(prov_indices)} 样本")
    
    # 构建宏区 folds
    print("\n🔀 创建宏区宽松 Out-of-Region 划分...")
    folds = []
    all_provinces = set(province_samples.keys())
    
    for fold_id, (macro_name, prov_list) in enumerate(MACRO_REGION_DEF.items(), 1):
        # 只保留实际存在的省份
        test_provinces = [p for p in prov_list if p in province_samples]
        if len(test_provinces) == 0:
            print(f"\n   ⚠️ 宏区 {macro_name} 在数据中没有任何省份，跳过")
            continue
        
        trainval_provinces = sorted(list(all_provinces - set(test_provinces)))
        
        # 该宏区所有样本
        region_idx_all = []
        for p in test_provinces:
            region_idx_all.extend(province_samples[p])
        region_idx_all = np.array(region_idx_all, dtype=int)
        
        # 其他所有宏区样本
        other_idx_all = []
        for p in trainval_provinces:
            other_idx_all.extend(province_samples[p])
        other_idx_all = np.array(other_idx_all, dtype=int)
        
        # 在该宏区内部划分出 test / pool_target
        rng_region = np.random.RandomState(seed=fold_id * 100 + 7)
        rng_region.shuffle(region_idx_all)
        n_region = len(region_idx_all)
        n_test_target = max(1, int(n_region * target_test_ratio))
        test_idx = region_idx_all[:n_test_target]
        pool_target = region_idx_all[n_test_target:]
        
        # train+val 池 = 其他宏区 + 目标宏区剩余样本
        trainval_pool = np.concatenate([other_idx_all, pool_target], axis=0)
        rng_tv = np.random.RandomState(seed=fold_id * 1000 + 13)
        rng_tv.shuffle(trainval_pool)
        
        n_tv = len(trainval_pool)
        n_val = max(1, int(n_tv * global_val_ratio))
        val_idx = trainval_pool[:n_val]
        train_idx = trainval_pool[n_val:]
        
        print(f"\n   宏区 {macro_name}:")
        print(f"      测试省份 ({len(test_provinces)}): {', '.join(test_provinces)}")
        print(f"      训练+验证省份 ({len(trainval_provinces)}): {', '.join(trainval_provinces[:10])}...")
        print(f"      目标宏区样本总数: {n_region}")
        print(f"         ➜ 测试集: {len(test_idx)}")
        print(f"         ➜ 进入 train+val 池: {len(pool_target)}")
        print(f"      其他宏区样本数: {len(other_idx_all)}")
        print(f"      最终 train 样本: {len(train_idx)}")
        print(f"      最终 val 样本: {len(val_idx)}")
        print(f"      最终 test 样本: {len(test_idx)}")
        
        folds.append({
            "fold": fold_id,
            "macro_region": macro_name,
            "train_provinces": trainval_provinces,  # 训练+验证来源的省
            "test_provinces": test_provinces,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
        })
    
    print("\n✅ 宏区划分完成")
    return folds, province_samples

# ============================================================================
#               第五部分：检查点管理功能（保持原样）
# ============================================================================

class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, fold_id, checkpoint_dir):
        self.fold_id = fold_id
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = os.path.join(
            checkpoint_dir, 
            f"fold{fold_id}_checkpoint.pth"
        )
        self.meta_path = os.path.join(
            checkpoint_dir,
            f"fold{fold_id}_meta.json"
        )
    
    def save_checkpoint(self, epoch, model, optimizer, scheduler, 
                       best_val_r2, best_state, epochs_no_improve):
        """保存训练检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_r2': best_val_r2,
            'best_state': best_state,  # 保存最优模型状态
            'epochs_no_improve': epochs_no_improve,
            'fold_id': self.fold_id,
        }
        
        try:
            temp_path = self.checkpoint_path + '.tmp'
            torch.save(checkpoint, temp_path)
            
            if os.path.exists(self.checkpoint_path):
                os.remove(self.checkpoint_path)
            
            os.rename(temp_path, self.checkpoint_path)
            
            meta = {
                'epoch': epoch,
                'best_val_r2': float(best_val_r2),
                'epochs_no_improve': epochs_no_improve,
                'fold_id': self.fold_id,
                'timestamp': time.time(),
            }
            with open(self.meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            
            return True
        except Exception as e:
            print(f"\n   ⚠️ 检查点保存失败: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            return False
    
    def load_checkpoint(self, model, optimizer, scheduler):
        """加载训练检查点"""
        if not os.path.exists(self.checkpoint_path):
            return None
        
        try:
            print(f"\n   🔄 发现检查点，正在加载...")
            checkpoint = torch.load(self.checkpoint_path, map_location=DEVICE)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            start_epoch = checkpoint['epoch'] + 1
            best_val_r2 = checkpoint['best_val_r2']
            best_state = checkpoint.get('best_state', None)
            epochs_no_improve = checkpoint['epochs_no_improve']
            
            print(f"   ✅ 从 Epoch {checkpoint['epoch']} 恢复")
            print(f"   最佳验证R²: {best_val_r2:.4f}")
            print(f"   未改善轮数: {epochs_no_improve}")
            
            return {
                'start_epoch': start_epoch,
                'best_val_r2': best_val_r2,
                'best_state': best_state,
                'epochs_no_improve': epochs_no_improve,
            }
        except Exception as e:
            print(f"\n   ⚠️ 检查点加载失败: {e}")
            print(f"   将从头开始训练")
            return None
    
    def clean(self):
        """清理检查点"""
        try:
            if os.path.exists(self.checkpoint_path):
                os.remove(self.checkpoint_path)
                print(f"   🗑️  已清理检查点: {os.path.basename(self.checkpoint_path)}")
            if os.path.exists(self.meta_path):
                os.remove(self.meta_path)
        except Exception as e:
            print(f"   ⚠️ 检查点清理失败: {e}")

# ============================================================================
#        第六部分：单 fold 训练函数（加入宏区名称，但其它逻辑不变）
# ============================================================================

def train_one_fold_oop(base_ds, train_idx, val_idx, test_idx, 
                       fold_id, train_provinces, test_provinces,
                       macro_region=None):
    """训练单个 fold（支持断点续跑，宽松版 Out-of-Region）"""
    print(f"\n{'='*80}")
    if macro_region is not None:
        print(f"🎯 Fold {fold_id} - Out-of-Region 宏区训练（测试宏区: {macro_region}）")
    else:
        print(f"🎯 Fold {fold_id} - Out-of-Region 宏区训练")
    print(f"{'='*80}")
    if macro_region is not None:
        print(f"   测试宏区: {macro_region}")
    print(f"   训练省份数: {len(train_provinces)}")
    print(f"   测试省份数: {len(test_provinces)}")
    print(f"   训练样本: {len(train_idx):,}")
    print(f"   验证样本: {len(val_idx):,}")
    print(f"   测试样本: {len(test_idx):,}")
    
    # 创建数据集
    train_subset = Subset(base_ds, list(train_idx))
    val_subset = Subset(base_ds, list(val_idx))
    test_subset = Subset(base_ds, list(test_idx))
    
    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    val_loader = DataLoader(
        val_subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    test_loader = DataLoader(
        test_subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    # 模型
    model = FusionResNetDirichlet(
        day_channels=len(DAY_TIFS), night_channels=1,
        n_comp=len(TARGET_FIELDS),
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=BEST_LR, weight_decay=BEST_WD
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-7
    )
    
    # 检查点管理器
    checkpoint_mgr = CheckpointManager(fold_id, CHECKPOINT_DIR)
    
    # 尝试加载检查点
    checkpoint_info = checkpoint_mgr.load_checkpoint(model, optimizer, scheduler)
    
    if checkpoint_info is not None:
        start_epoch = checkpoint_info['start_epoch']
        best_val_r2 = checkpoint_info['best_val_r2']
        best_state = checkpoint_info['best_state']
        epochs_no_improve = checkpoint_info['epochs_no_improve']
    else:
        start_epoch = 1
        best_val_r2 = -1e9
        best_state = None
        epochs_no_improve = 0
    
    print(f"\n⏳ 开始训练（从 Epoch {start_epoch} 开始）...")
    pbar = tqdm(range(start_epoch, EPOCHS + 1), desc=f"Fold{fold_id}", 
                initial=start_epoch-1, total=EPOCHS)
    
    for epoch in pbar:
        # 训练
        base_ds.set_mode('train')
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            day = batch["day"].to(DEVICE, non_blocking=True)
            night = batch["night"].to(DEVICE, non_blocking=True)
            y = batch["y"].to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            alpha = model(day, night)
            loss = dirichlet_nll(y, alpha)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * day.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # 验证
        base_ds.set_mode('val')
        model.eval()
        val_loss = 0.0
        y_true_all = []
        y_pred_all = []
        with torch.no_grad():
            for batch in val_loader:
                day = batch["day"].to(DEVICE, non_blocking=True)
                night = batch["night"].to(DEVICE, non_blocking=True)
                y = batch["y"].to(DEVICE, non_blocking=True)
                
                alpha = model(day, night)
                loss = dirichlet_nll(y, alpha)
                val_loss += loss.item() * day.size(0)
                
                B, C, H, W = alpha.shape
                ac = alpha[:, :, H // 2, W // 2].cpu().numpy()
                pred_comp = ac / np.clip(ac.sum(axis=1, keepdims=True), 1e-6, None)
                
                y_np = y.cpu().numpy()
                y_true_all.append(y_np)
                y_pred_all.append(pred_comp)
        
        val_loss /= len(val_loader.dataset)
        y_true_all = np.vstack(y_true_all)
        y_pred_all = np.vstack(y_pred_all)
        val_r2_mean, _ = r2_score_numpy(y_true_all, y_pred_all)
        
        scheduler.step(val_r2_mean)
        current_lr = optimizer.param_groups[0]['lr']
        
        pbar.set_postfix({
            'loss': f'{train_loss:.3f}',
            'val_r2': f'{val_r2_mean:.3f}',
            'best': f'{best_val_r2:.3f}',
            'lr': f'{current_lr:.2e}'
        })
        
        # 更新最优模型
        if val_r2_mean > best_val_r2 + 1e-4:
            best_val_r2 = val_r2_mean
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # 定期保存检查点
        if epoch % CHECKPOINT_INTERVAL == 0:
            checkpoint_mgr.save_checkpoint(
                epoch, model, optimizer, scheduler,
                best_val_r2, best_state, epochs_no_improve
            )
        
        # 早停
        if epochs_no_improve >= PATIENCE:
            pbar.set_description(f"Fold{fold_id} [早停]")
            break
    
    # 训练完成后清理检查点
    checkpoint_mgr.clean()
    
    # 测试
    if best_state is not None:
        model.load_state_dict(best_state)
    
    base_ds.set_mode('val')
    model.eval()
    y_true_all = []
    y_pred_all = []
    with torch.no_grad():
        for batch in test_loader:
            day = batch["day"].to(DEVICE, non_blocking=True)
            night = batch["night"].to(DEVICE, non_blocking=True)
            y = batch["y"].to(DEVICE, non_blocking=True)
            
            alpha = model(day, night)
            B, C, H, W = alpha.shape
            ac = alpha[:, :, H // 2, W // 2].cpu().numpy()
            pred_comp = ac / np.clip(ac.sum(axis=1, keepdims=True), 1e-6, None)
            
            y_np = y.cpu().numpy()
            y_true_all.append(y_np)
            y_pred_all.append(pred_comp)
    
    y_true_all = np.vstack(y_true_all)
    y_pred_all = np.vstack(y_pred_all)
    test_r2_mean, test_r2_each = r2_score_numpy(y_true_all, y_pred_all)
    
    # 保存最终模型
    model_path = os.path.join(
        OUT_DIR,
        f"model_OOR_macro_soft_fold{fold_id}_lr{BEST_LR:g}_wd{BEST_WD:g}.pth"
    )
    torch.save(best_state, model_path)
    
    print(f"\n✅ Fold {fold_id} 完成:")
    print(f"   验证R²: {best_val_r2:.4f}")
    print(f"   测试R²: {test_r2_mean:.4f}")
    print(f"   模型保存: {model_path}")
    
    metrics = {
        "fold": fold_id,
        "macro_region": macro_region if macro_region is not None else "",
        "train_provinces": ", ".join(train_provinces),
        "test_provinces": ", ".join(test_provinces),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "val_r2_best": best_val_r2,
        "test_r2_mean": test_r2_mean,
    }
    
    for name, r2v in zip(TARGET_FIELDS, test_r2_each):
        metrics[f"test_r2_{name}"] = r2v
    
    metrics["model_path"] = model_path
    
    return metrics

# ============================================================================
#               第七部分：主程序（支持跳过已完成 fold）
# ============================================================================

def main():
    """主程序（宽松版 Out-of-Region 宏区划分 + 断点续跑）"""
    print("\n" + "="*80)
    print("🚀 开始 Out-of-Region 宏区验证训练（宽松版）")
    print("="*80)
    
    results_csv = os.path.join(OUT_DIR, "out_of_region_macro_soft_results.csv")
    
    # 已完成的 fold
    completed_folds = []
    all_results = []
    
    if os.path.exists(results_csv):
        try:
            df_existing = pd.read_csv(results_csv)
            if 'fold' in df_existing.columns:
                completed_folds = df_existing['fold'].tolist()
            all_results = df_existing.to_dict('records')
            print(f"\n📊 发现已完成的fold: {completed_folds}")
            print(f"   已加载 {len(all_results)} 个结果")
        except Exception as e:
            print(f"\n⚠️ 读取已有结果失败: {e}")
    
    # 1. 构建数据集
    print("\n📊 构建基础数据集...")
    base_ds = PatchPointDataset(
        LABEL_SHP, DAY_TIFS, NIGHT_TIF, TARGET_FIELDS,
        patch_size=PATCH_SIZE, mode='val', check_valid=True, 
        max_samples=None, cache_data=True
    )
    
    # 2. 宏区划分（宽松版 Out-of-Region）
    folds, province_samples = create_macroregion_folds_soft(
        base_ds, PROVINCE_SHP,
        target_test_ratio=TARGET_TEST_RATIO,
        global_val_ratio=GLOBAL_VAL_RATIO
    )
    
    # 3. 逐 fold 训练（跳过已完成的）
    for fold_info in folds:
        fold_id = fold_info["fold"]
        macro_region = fold_info["macro_region"]
        
        if fold_id in completed_folds:
            print(f"\n{'='*80}")
            print(f"⏭️  Fold {fold_id} (宏区: {macro_region}) 已完成，跳过")
            print(f"{'='*80}")
            continue
        
        train_idx = fold_info["train_idx"]
        val_idx = fold_info["val_idx"]
        test_idx = fold_info["test_idx"]
        train_provinces = fold_info["train_provinces"]
        test_provinces = fold_info["test_provinces"]
        
        try:
            metrics = train_one_fold_oop(
                base_ds, train_idx, val_idx, test_idx,
                fold_id, train_provinces, test_provinces,
                macro_region=macro_region
            )
            
            # 更新 / 追加结果
            existing_fold_idx = None
            for i, result in enumerate(all_results):
                if result.get('fold', None) == fold_id:
                    existing_fold_idx = i
                    break
            
            if existing_fold_idx is not None:
                all_results[existing_fold_idx] = metrics
            else:
                all_results.append(metrics)
            
            df_results = pd.DataFrame(all_results)
            df_results.to_csv(results_csv, index=False)
            print(f"\n💾 结果已保存到: {results_csv}")
            
        except KeyboardInterrupt:
            print(f"\n\n⏹️  用户中断训练（Ctrl+C）")
            print(f"   Fold {fold_id} 的检查点已保存")
            print(f"   重新运行脚本将从当前进度继续")
            base_ds.close()
            return
        
        except Exception as e:
            print(f"\n❌ Fold {fold_id} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            
            error_result = {
                "fold": fold_id,
                "macro_region": macro_region,
                "train_provinces": ", ".join(train_provinces),
                "test_provinces": ", ".join(test_provinces),
                "error": str(e),
            }
            all_results.append(error_result)
            
            df_results = pd.DataFrame(all_results)
            df_results.to_csv(results_csv, index=False)
            continue
    
    # 4. 最终总结
    print("\n" + "="*80)
    print("✅ Out-of-Region 宏区验证（宽松版）完成！")
    print("="*80)
    
    df_results = pd.DataFrame(all_results)
    
    if 'error' in df_results.columns:
        df_success = df_results[df_results['error'].isna()]
    else:
        df_success = df_results
    
    if not df_success.empty and 'test_r2_mean' in df_success.columns:
        print("\n📊 总结:")
        print(f"   平均测试R²: {df_success['test_r2_mean'].mean():.4f} ± {df_success['test_r2_mean'].std():.4f}")
        print(f"\n   各fold结果:")
        for _, row in df_success.iterrows():
            test_provs = row['test_provinces']
            if isinstance(test_provs, str) and len(test_provs) > 60:
                test_provs_disp = test_provs[:60] + "..."
            else:
                test_provs_disp = test_provs
            mr = row.get('macro_region', '')
            print(f"      Fold{int(row['fold'])} (宏区: {mr}): "
                  f"R²={row['test_r2_mean']:.4f} (测试省份: {test_provs_disp})")
    
    print(f"\n   结果保存: {results_csv}")
    print(f"   模型保存: {OUT_DIR}")
    
    # 清理空的检查点目录
    try:
        if os.path.exists(CHECKPOINT_DIR) and not os.listdir(CHECKPOINT_DIR):
            os.rmdir(CHECKPOINT_DIR)
            print(f"\n🗑️  已清理空检查点目录")
    except:
        pass
    
    base_ds.close()


if __name__ == "__main__":
    main()
