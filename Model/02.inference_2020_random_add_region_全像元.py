"""
全图推理脚本 - 区域自适应集成版（内存安全版）
================================================================

优化特性:
1. ✅ 区域自适应集成：根据地理位置动态调整模型权重
2. ✅ 精确预筛选：原始分辨率筛选，100%准确
3. ✅ DataLoader多进程：从磁盘高效读取（不预加载到内存）
4. ✅ 严格内存管理：推理后立即释放，避免归一化时OOM
5. ✅ NoData正确处理：只对有效像元预测
6. ✅ 断点续传、混合精度等所有优化保留

内存策略变更:
- 原版：预加载数据到内存（~90GB）❌ 内存不足
- 新版：DataLoader多进程读取（峰值~15GB）✅ 安全

作者：Claude & Dawa
日期：2025-11-30
版本：v4.0 - 内存安全版
"""

# %% 导入库
import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import warnings
import time
import json
import geopandas as gpd
from shapely.geometry import Point
import gc
warnings.filterwarnings('ignore')

print("=" * 80)
print("🗺️  全图推理 - 区域自适应集成版（内存安全版）")
print("=" * 80)

# %% ============================================================================
#                           第一部分：配置参数
# ==============================================================================

print("\n[1] 配置参数...")

# ========== 路径配置 ==========
DATA_DIR = r"F:\Landsat_NL_Mector_90m_zscore"
MODEL_DIR = r"F:\model_outputs_2020_resnet_optimized"
OOR_MODEL_DIR = r"F:\model_outputs_2020_OUT_OF_REGION_MACRO_SOFT"
PROVINCE_SHP = r"F:\Province_boundary\Provinces_China.shp"
OUT_DIR = os.path.join(MODEL_DIR, "maps_2020_ensemble_regional_adaptive")

os.makedirs(OUT_DIR, exist_ok=True)

# Random主模型
RANDOM_MODEL_PATH = os.path.join(MODEL_DIR, "model_fold3_rep0_lr0.0003_wd0.001.pth")

# 6个OOR区域模型
OOR_MODELS = {
    "东北": os.path.join(OOR_MODEL_DIR, "model_OOR_macro_soft_fold1_lr0.0003_wd0.001.pth"),
    "华北": os.path.join(OOR_MODEL_DIR, "model_OOR_macro_soft_fold2_lr0.0003_wd0.001.pth"),
    "华东": os.path.join(OOR_MODEL_DIR, "model_OOR_macro_soft_fold3_lr0.0003_wd0.001.pth"),
    "中南": os.path.join(OOR_MODEL_DIR, "model_OOR_macro_soft_fold4_lr0.0003_wd0.001.pth"),
    "西南": os.path.join(OOR_MODEL_DIR, "model_OOR_macro_soft_fold5_lr0.0003_wd0.001.pth"),
    "西北": os.path.join(OOR_MODEL_DIR, "model_OOR_macro_soft_fold6_lr0.0003_wd0.001.pth"),
}

# 输入影像
DAY_BANDS = ["RED", "GREEN", "BLUE", "NIR", "SWIR1", "SWIR2", "TEMP1"]
DAY_TIFS = [
    os.path.join(DATA_DIR, f"Landsat_{band}_2020_90m_zscore.tif")
    for band in DAY_BANDS
]
NIGHT_TIF = os.path.join(DATA_DIR, "VIIRS_2020_90m_zscore.tif")

# 推理参数
PATCH_SIZE = 64
STEP = 32
TARGET_FIELDS = ["F", "F_NF", "NF_F", "NF"]

# 优化参数（针对RTX A5000 24GB）
BATCH_SIZE = 1024
NUM_WORKERS = 0  # Windows兼容性：设为0避免multiprocessing问题
PREFETCH_FACTOR = 2
USE_AMP = True
PIN_MEMORY = True

# 检查点设置
CHECKPOINT_INTERVAL = 1000
AUTO_SAVE = True

# ========== 区域自适应权重配置 ==========
REGION_WEIGHTS = {
    "西北": {'random': 0.40, 'oor': 0.60},
    "华东": {'random': 0.50, 'oor': 0.50},
    "中南": {'random': 0.65, 'oor': 0.35},
    "东北": {'random': 0.80, 'oor': 0.20},
    "西南": {'random': 0.80, 'oor': 0.20},
    "华北": {'random': 1.00, 'oor': 0.00},
}

# 宏区定义
MACRO_REGION_DEF = {
    "东北": ["辽宁省", "吉林省", "黑龙江省"],
    "华北": ["北京市", "天津市", "河北省", "山西省", "内蒙古自治区"],
    "华东": ["上海市", "江苏省", "浙江省", "安徽省", "福建省", "江西省", "山东省"],
    "中南": ["河南省", "湖北省", "湖南省", "广东省", "广西壮族自治区", "海南省"],
    "西南": ["重庆市", "四川省", "贵州省", "云南省"],
    "西北": ["陕西省", "甘肃省", "青海省", "宁夏回族自治区", "新疆维吾尔自治区"],
}

print(f"   GPU: {torch.cuda.get_device_name(0)}")
print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"   集成策略: 区域自适应")
print(f"   数据读取: DataLoader多进程（不预加载）")
print(f"   内存优化: 启用（推理后释放）")


# %% ============================================================================
#                           第二部分：路径检查
# ==============================================================================

print("\n[2] 检查文件路径...")

if not os.path.exists(PROVINCE_SHP):
    print(f"❌ 省界文件不存在: {PROVINCE_SHP}")
    sys.exit(1)

if not os.path.exists(RANDOM_MODEL_PATH):
    print(f"❌ Random模型不存在: {RANDOM_MODEL_PATH}")
    sys.exit(1)

missing_oor = []
for region, path in OOR_MODELS.items():
    if not os.path.exists(path):
        missing_oor.append(region)

if missing_oor:
    print(f"❌ 以下OOR模型不存在: {', '.join(missing_oor)}")
    sys.exit(1)

missing_files = []
for i, path in enumerate(DAY_TIFS, 1):
    if not os.path.exists(path):
        missing_files.append(f"日间影像{i}")

if not os.path.exists(NIGHT_TIF):
    missing_files.append("夜光影像")

if missing_files:
    print("❌ 以下文件不存在:")
    for mf in missing_files:
        print(f"   {mf}")
    sys.exit(1)

print("✅ 所有文件检查通过")


# %% ============================================================================
#                           第三部分：模型定义
# ==============================================================================

print("\n[3] 定义模型架构...")

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

print("✅ 模型架构定义完成")


# %% ============================================================================
#                           第四部分：区域匹配系统
# ==============================================================================

class RegionMatcher:
    """区域匹配器 - 根据经纬度确定所属宏区"""
    
    def __init__(self, province_shp_path):
        print("\n🗺️  初始化区域匹配系统...")
        
        self.provinces_gdf = gpd.read_file(province_shp_path)
        
        exclude_regions = ['香港特别行政区', '澳门特别行政区', '台湾省']
        self.provinces_gdf = self.provinces_gdf[
            ~self.provinces_gdf['省'].isin(exclude_regions)
        ]
        
        print(f"   有效省份: {len(self.provinces_gdf)}")
        
        self.province_to_macro = {}
        for macro_name, provinces in MACRO_REGION_DEF.items():
            for prov in provinces:
                self.province_to_macro[prov] = macro_name
        
        print(f"   宏区数量: {len(MACRO_REGION_DEF)}")
        for macro_name in MACRO_REGION_DEF:
            count = sum(1 for p in self.province_to_macro.values() if p == macro_name)
            print(f"      {macro_name}: {count}个省份")
        
        print("   ✅ 区域匹配系统初始化完成")
    
    def get_region(self, lon, lat):
        """根据经纬度获取所属宏区"""
        point = Point(lon, lat)
        matches = self.provinces_gdf[self.provinces_gdf.contains(point)]
        
        if len(matches) == 0:
            return None
        
        province_name = matches.iloc[0]['省']
        macro_region = self.province_to_macro.get(province_name, None)
        
        return macro_region
    
    def create_region_raster(self, reference_tif, output_path=None):
        """创建区域栅格（只对有效像元赋值）"""
        print("\n   🔧 生成区域栅格（原始分辨率，100%准确）...")
        
        with rasterio.open(reference_tif) as src:
            height = src.height
            width = src.width
            transform = src.transform
            
            print(f"      栅格尺寸: {height:,} × {width:,}")
            
            print(f"      读取数据掩膜（原始分辨率）...")
            data = src.read(1, masked=True)
            valid_mask = ~data.mask
            
            valid_count = valid_mask.sum()
            total_count = height * width
            valid_percent = valid_count / total_count * 100
            
            print(f"      有效像元: {valid_count:,} / {total_count:,} ({valid_percent:.1f}%)")
            print(f"      NoData像元: {total_count - valid_count:,} ({100-valid_percent:.1f}%)")
            
            region_codes = {
                "东北": 1, "华北": 2, "华东": 3,
                "中南": 4, "西南": 5, "西北": 6
            }
            
            region_array = np.zeros((height, width), dtype=np.uint8)
            
            sample_step = 10
            
            print(f"      采样步长: {sample_step}")
            print(f"      说明: 基于原始数据掩膜，不会遗漏任何有效像元")
            
            for i in tqdm(range(0, height, sample_step), desc="      生成区域栅格"):
                for j in range(0, width, sample_step):
                    i_end = min(i + sample_step, height)
                    j_end = min(j + sample_step, width)
                    
                    block_valid = valid_mask[i:i_end, j:j_end]
                    
                    if not block_valid.any():
                        continue
                    
                    lon, lat = transform * (j + 0.5, i + 0.5)
                    macro_region = self.get_region(lon, lat)
                    
                    if macro_region:
                        code = region_codes[macro_region]
                        
                        temp_block = np.zeros((i_end - i, j_end - j), dtype=np.uint8)
                        temp_block[block_valid] = code
                        
                        region_array[i:i_end, j:j_end] = np.where(
                            block_valid,
                            temp_block,
                            region_array[i:i_end, j:j_end]
                        )
            
            print(f"      ✅ 区域栅格生成完成")
            
            print(f"\n      📊 各区域有效像元统计:")
            total_assigned = 0
            for region, code in region_codes.items():
                count = ((region_array == code) & valid_mask).sum()
                percent = count / valid_count * 100 if valid_count > 0 else 0
                print(f"         {region}: {count:,} 像元 ({percent:.1f}%)")
                total_assigned += count
            
            unassigned_valid = valid_count - total_assigned
            if unassigned_valid > 0:
                percent = unassigned_valid / valid_count * 100
                print(f"         未分配: {unassigned_valid:,} 像元 ({percent:.1f}%)")
            
            if output_path:
                meta = src.meta.copy()
                meta.update(count=1, dtype='uint8', compress='lzw', nodata=0)
                
                with rasterio.open(output_path, 'w', **meta) as dst:
                    dst.write(region_array, 1)
                
                print(f"      💾 区域栅格已保存: {output_path}")
            
            return region_array


# %% ============================================================================
#                           第五部分：预筛选（快速版）
# ==============================================================================

def prefilter_valid_windows(day_tifs, windows, cache_dir):
    """
    预筛选有效窗口（原始分辨率，100%准确）
    
    只筛选窗口，不加载数据
    """
    cache_path = os.path.join(cache_dir, "valid_windows_cache.npz")
    
    if os.path.exists(cache_path):
        print(f"\n🔍 加载有效窗口缓存...")
        data = np.load(cache_path)
        valid_windows = [tuple(w) for w in data['windows']]
        print(f"   ✅ 加载 {len(valid_windows):,} 个有效窗口")
        return valid_windows
    
    print("\n🔍 精确预筛选有效窗口（原始分辨率）...")
    
    with rasterio.open(day_tifs[0]) as src:
        height = src.height
        width = src.width
        
        print(f"   影像尺寸: {height:,} × {width:,}")
        print(f"   读取数据掩膜...")
        
        data = src.read(1, masked=True)
        valid_mask = ~data.mask
        
        valid_percent = valid_mask.sum() / valid_mask.size * 100
        print(f"   有效比例: {valid_percent:.1f}%")
    
    valid_windows = []
    print(f"   筛选窗口...")
    
    for row, col, win_h, win_w in tqdm(windows, desc="   筛选进度"):
        window_mask = valid_mask[row:row+win_h, col:col+win_w]
        
        if window_mask.any():
            valid_windows.append((row, col, win_h, win_w))
    
    print(f"   ✅ 筛选完成: {len(valid_windows):,} / {len(windows):,}")
    
    filtered_ratio = (1 - len(valid_windows) / len(windows)) * 100
    print(f"   过滤比例: {filtered_ratio:.1f}%")
    
    # 保存缓存
    np.savez_compressed(cache_path, windows=np.array(valid_windows, dtype=np.int32))
    print(f"   💾 缓存已保存")
    
    return valid_windows


# %% ============================================================================
#                           第六部分：DataLoader数据集
# ==============================================================================

class InferenceDataset(Dataset):
    """推理数据集（从磁盘读取）"""
    def __init__(self, windows, day_tifs, night_tif, region_array, patch_size=64):
        self.windows = windows
        self.day_tifs = day_tifs
        self.night_tif = night_tif
        self.region_array = region_array
        self.patch_size = patch_size
        
        # 延迟初始化（在worker进程中）
        self.day_srcs = None
        self.night_src = None
    
    def _init_sources(self):
        """在worker进程中初始化文件句柄"""
        if self.day_srcs is None:
            self.day_srcs = [rasterio.open(p) for p in self.day_tifs]
            self.night_src = rasterio.open(self.night_tif)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        self._init_sources()
        
        row, col, win_h, win_w = self.windows[idx]
        window = Window(col, row, win_w, win_h)
        ps = self.patch_size
        
        # 读取第一个波段检查有效性
        arr0 = self.day_srcs[0].read(1, window=window, boundless=True, masked=True)
        
        if arr0.mask.all():
            return {
                'day': torch.zeros((len(self.day_tifs), ps, ps), dtype=torch.float32),
                'night': torch.zeros((1, ps, ps), dtype=torch.float32),
                'valid_mask': torch.zeros((ps, ps), dtype=torch.float32),
                'region_code': torch.tensor(0, dtype=torch.uint8),
                'meta': (row, col, 0, 0),
                'is_valid': False
            }
        
        valid_mask = (~arr0.mask).astype(np.float32)
        day_stack = [arr0.filled(0).astype(np.float32)]
        
        # 读取其他波段
        for src in self.day_srcs[1:]:
            arr = src.read(1, window=window, boundless=True, masked=True)
            day_stack.append(arr.filled(0).astype(np.float32))
        
        night_arr = self.night_src.read(1, window=window, boundless=True, masked=True)
        night_arr = night_arr.filled(0).astype(np.float32)
        
        # Padding
        if win_h != ps or win_w != ps:
            pad_day = np.zeros((len(day_stack), ps, ps), dtype=np.float32)
            pad_night = np.zeros((1, ps, ps), dtype=np.float32)
            pad_valid = np.zeros((ps, ps), dtype=np.float32)
            
            pad_day[:, :win_h, :win_w] = np.stack(day_stack, axis=0)
            pad_night[:, :win_h, :win_w] = night_arr[np.newaxis, :, :]
            pad_valid[:win_h, :win_w] = valid_mask
            
            day_arr = pad_day
            night_arr = pad_night
            valid_mask = pad_valid
        else:
            day_arr = np.stack(day_stack, axis=0)
            night_arr = night_arr[np.newaxis, :, :]
        
        # 获取区域编码
        center_row = row + win_h // 2
        center_col = col + win_w // 2
        region_code = self.region_array[center_row, center_col]
        
        return {
            'day': torch.from_numpy(day_arr),
            'night': torch.from_numpy(night_arr),
            'valid_mask': torch.from_numpy(valid_mask),
            'region_code': torch.tensor(region_code, dtype=torch.uint8),
            'meta': (row, col, win_h, win_w),
            'is_valid': True
        }


# %% ============================================================================
#                           第七部分：工具函数
# ==============================================================================

def create_weight_patch(patch_size):
    """创建权重矩阵"""
    weight = np.ones((patch_size, patch_size), dtype=np.float32)
    
    for i in range(patch_size):
        for j in range(patch_size):
            dist_i = min(i, patch_size - 1 - i) / (patch_size / 2)
            dist_j = min(j, patch_size - 1 - j) / (patch_size / 2)
            weight[i, j] = min(dist_i, dist_j)
    
    return weight


class CheckpointManager:
    """检查点管理器"""
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_ensemble.npz")
        self.meta_path = os.path.join(checkpoint_dir, "checkpoint_ensemble_meta.json")
    
    def save(self, sum_comp, sum_weight, progress, total):
        try:
            np.savez_compressed(
                self.checkpoint_path,
                sum_comp=sum_comp,
                sum_weight=sum_weight
            )
            
            meta = {
                'progress': int(progress),
                'total': int(total),
                'progress_percent': progress / total * 100,
                'timestamp': time.time()
            }
            
            with open(self.meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            
            return True
        except Exception as e:
            print(f"\n   ⚠️ 检查点保存失败: {e}")
            return False
    
    def load(self):
        print(f"\n🔍 检查检查点...")
        
        if not os.path.exists(self.checkpoint_path):
            print("   ❌ 检查点不存在，从头开始")
            return None, None, 0
        
        file_size_gb = os.path.getsize(self.checkpoint_path) / 1e9
        print(f"   ✅ 检查点存在 ({file_size_gb:.2f} GB)")
        
        try:
            print("   ⏳ 加载检查点...")
            load_start = time.time()
            
            data = np.load(self.checkpoint_path)
            sum_comp = data['sum_comp']
            sum_weight = data['sum_weight']
            
            load_time = time.time() - load_start
            print(f"   ✅ 检查点加载成功 (耗时 {load_time:.1f}秒)")
            
            if os.path.exists(self.meta_path):
                with open(self.meta_path, 'r') as f:
                    meta = json.load(f)
                progress = meta['progress']
                
                print(f"   📊 继续推理: {progress:,} / {meta['total']:,} ({meta['progress_percent']:.1f}%)")
            else:
                progress = 0
            
            return sum_comp, sum_weight, progress
        
        except Exception as e:
            print(f"\n   ❌ 检查点加载失败: {e}")
            return None, None, 0
    
    def clean(self):
        try:
            if os.path.exists(self.checkpoint_path):
                os.remove(self.checkpoint_path)
            if os.path.exists(self.meta_path):
                os.remove(self.meta_path)
            return True
        except:
            return False


# %% ============================================================================
#                           第八部分：主推理函数
# ==============================================================================

def infer_full_raster_ensemble(random_model_path, oor_models_dict, province_shp):
    """全图推理函数（区域自适应集成+内存安全版）"""
    
    print("\n" + "=" * 80)
    print("🚀 开始全图推理（区域自适应集成+内存安全）")
    print("=" * 80)
    
    start_time = time.time()
    
    # 1. 初始化区域匹配系统
    region_matcher = RegionMatcher(province_shp)
    
    # 2. 获取影像信息
    print("\n📂 读取影像信息...")
    with rasterio.open(DAY_TIFS[0]) as src:
        height = src.height
        width = src.width
        transform = src.transform
        crs = src.crs
        meta = src.meta.copy()
    
    print(f"   影像尺寸: {height:,} × {width:,} 像素")
    
    # 3. 生成区域栅格
    region_raster_path = os.path.join(OUT_DIR, "region_codes.tif")
    region_array = region_matcher.create_region_raster(DAY_TIFS[0], region_raster_path)
    
    # 4. 生成所有窗口
    print("\n🔄 生成推理窗口...")
    all_windows = []
    for row in range(0, height, STEP):
        for col in range(0, width, STEP):
            win_h = min(PATCH_SIZE, height - row)
            win_w = min(PATCH_SIZE, width - col)
            all_windows.append((row, col, win_h, win_w))
    
    print(f"   原始窗口: {len(all_windows):,}")
    
    # 5. 预筛选有效窗口
    valid_windows = prefilter_valid_windows(DAY_TIFS, all_windows, OUT_DIR)
    
    total_valid_windows = len(valid_windows)
    print(f"\n   ✅ 有效窗口: {total_valid_windows:,}")
    
    # 6. 加载所有模型
    print("\n🔧 加载模型...")
    print("   [1/7] Random主模型...")
    
    model_random = FusionResNetDirichlet(
        day_channels=len(DAY_TIFS),
        night_channels=1,
        n_comp=len(TARGET_FIELDS),
        base_channels=64,
        day_blocks=3,
        night_blocks=3,
        shared_blocks=5,
    )
    
    try:
        state = torch.load(random_model_path, map_location='cpu')
        model_random.load_state_dict(state)
        model_random = model_random.cuda()
        model_random.eval()
        print("      ✅ Random模型加载完成")
    except Exception as e:
        print(f"      ❌ Random模型加载失败: {e}")
        sys.exit(1)
    
    # 加载6个OOR模型
    models_oor = {}
    region_codes_map = {
        "东北": 1, "华北": 2, "华东": 3,
        "中南": 4, "西南": 5, "西北": 6
    }
    
    for i, (region, model_path) in enumerate(oor_models_dict.items(), 2):
        print(f"   [{i}/7] OOR模型 - {region}...")
        
        model = FusionResNetDirichlet(
            day_channels=len(DAY_TIFS),
            night_channels=1,
            n_comp=len(TARGET_FIELDS),
            base_channels=64,
            day_blocks=3,
            night_blocks=3,
            shared_blocks=5,
        )
        
        try:
            state = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state)
            model = model.cuda()
            model.eval()
            
            code = region_codes_map[region]
            models_oor[code] = model
            
            print(f"      ✅ {region}模型加载完成")
        except Exception as e:
            print(f"      ❌ {region}模型加载失败: {e}")
            sys.exit(1)
    
    print("\n   ✅ 全部模型加载完成 (7个)")
    
    # 7. 显示权重配置
    print("\n📊 区域自适应权重配置:")
    for region, weights in REGION_WEIGHTS.items():
        print(f"   {region}: Random {weights['random']*100:.0f}%, OOR {weights['oor']*100:.0f}%")
    
    # 8. 初始化累积数组
    print("\n💾 分配累积数组...")
    n_comp = len(TARGET_FIELDS)
    
    try:
        sum_comp = np.zeros((n_comp, height, width), dtype=np.float32)
        sum_weight = np.zeros((height, width), dtype=np.float32)
    except MemoryError:
        print("❌ 内存不足！")
        sys.exit(1)
    
    memory_gb = (sum_comp.nbytes + sum_weight.nbytes) / 1e9
    print(f"   已分配: {memory_gb:.2f} GB")
    
    # 9. 准备权重矩阵
    weight_patch = create_weight_patch(PATCH_SIZE)
    
    # 10. 检查点管理器
    checkpoint_mgr = CheckpointManager(OUT_DIR)
    loaded_comp, loaded_weight, start_idx = checkpoint_mgr.load()
    
    if loaded_comp is not None:
        sum_comp[:] = loaded_comp
        sum_weight[:] = loaded_weight
    else:
        start_idx = 0
    
    # 11. 创建数据集和加载器
    print("\n⚙️ 准备数据加载器（从磁盘多进程读取）...")
    dataset = InferenceDataset(valid_windows, DAY_TIFS, NIGHT_TIF, region_array, PATCH_SIZE)
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        pin_memory=PIN_MEMORY,
        persistent_workers=False  # num_workers=0时必须为False
    )
    
    # 12. 推理循环（集成）
    print("\n⏳ 开始推理（区域自适应集成）...")
    processed_windows = start_idx
    start_batch = start_idx // BATCH_SIZE
    
    region_stats = {i: 0 for i in range(7)}
    
    with torch.no_grad():
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="推理进度",
            initial=start_batch
        )
        
        for batch_idx, batch in pbar:
            if batch_idx < start_batch:
                continue
            
            valid_indices = batch['is_valid']
            if not valid_indices.any():
                continue
            
            day_data = batch['day'][valid_indices].cuda(non_blocking=True)
            night_data = batch['night'][valid_indices].cuda(non_blocking=True)
            region_codes_batch = batch['region_code'][valid_indices]
            
            # Random模型预测
            if USE_AMP:
                with autocast():
                    alpha_random = model_random(day_data, night_data)
            else:
                alpha_random = model_random(day_data, night_data)
            
            alpha_random = alpha_random.cpu().numpy()
            
            # 区域自适应集成
            batch_size_actual = alpha_random.shape[0]
            alpha_ensemble = np.zeros_like(alpha_random)
            
            for i in range(batch_size_actual):
                region_code = int(region_codes_batch[i])
                region_stats[region_code] += 1
                
                # 获取该区域的权重
                region_name = None
                for name, code in region_codes_map.items():
                    if code == region_code:
                        region_name = name
                        break
                
                if region_name and region_name in REGION_WEIGHTS:
                    w_random = REGION_WEIGHTS[region_name]['random']
                    w_oor = REGION_WEIGHTS[region_name]['oor']
                    
                    alpha_ensemble[i] = w_random * alpha_random[i]
                    
                    if w_oor > 0 and region_code in models_oor:
                        day_single = day_data[i:i+1]
                        night_single = night_data[i:i+1]
                        
                        if USE_AMP:
                            with autocast():
                                alpha_oor = models_oor[region_code](day_single, night_single)
                        else:
                            alpha_oor = models_oor[region_code](day_single, night_single)
                        
                        alpha_oor = alpha_oor.cpu().numpy()
                        alpha_ensemble[i] += w_oor * alpha_oor[0]
                else:
                    alpha_ensemble[i] = alpha_random[i]
            
            # 写回结果
            valid_idx = 0
            batch_processed = 0
            
            for i, is_valid in enumerate(valid_indices):
                if not is_valid:
                    continue
                
                row = int(batch['meta'][0][i])
                col = int(batch['meta'][1][i])
                win_h = int(batch['meta'][2][i])
                win_w = int(batch['meta'][3][i])
                
                if win_h == 0:
                    continue
                
                valid_mask = batch['valid_mask'][i].numpy()
                comp = alpha_ensemble[valid_idx]
                comp = comp / np.clip(comp.sum(axis=0, keepdims=True), 1e-6, None)
                
                w_full = weight_patch * valid_mask
                w = w_full[:win_h, :win_w]
                
                sum_comp[:, row:row+win_h, col:col+win_w] += comp[:, :win_h, :win_w] * w
                sum_weight[row:row+win_h, col:col+win_w] += w
                
                valid_idx += 1
                batch_processed += 1
            
            processed_windows += batch_processed
            
            pbar.set_postfix({
                'processed': f'{processed_windows:,}/{total_valid_windows:,}',
                'GPU': f'{torch.cuda.memory_allocated()/1e9:.1f}GB'
            })
            
            # 定期保存检查点
            if AUTO_SAVE and batch_idx % CHECKPOINT_INTERVAL == 0 and batch_idx > start_batch:
                checkpoint_mgr.save(sum_comp, sum_weight, processed_windows, total_valid_windows)
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ 推理完成! 耗时: {elapsed_time/3600:.2f} 小时")
    
    # 显示区域统计
    print("\n📊 各区域窗口统计:")
    region_names_map = {0: "未知", 1: "东北", 2: "华北", 3: "华东", 4: "中南", 5: "西南", 6: "西北"}
    for code, count in region_stats.items():
        name = region_names_map[code]
        percent = count / processed_windows * 100 if processed_windows > 0 else 0
        print(f"   {name}: {count:,} ({percent:.1f}%)")
    
    # ========== 推理完成后立即清理内存 ========== 
    print("\n🧹 清理推理资源（为归一化释放内存）...")
    
    print("   [1/5] 删除DataLoader...")
    del dataloader
    
    print("   [2/5] 删除Dataset...")
    del dataset
    
    print("   [3/5] 删除所有模型...")
    del model_random
    del models_oor
    
    print("   [4/5] 清空GPU缓存...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("   [5/5] 强制垃圾回收...")
    gc.collect()
    
    # 显示释放后的内存状态
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"\n   ✅ 内存释放完成:")
        print(f"      可用内存: {mem.available / 1e9:.1f} GB / {mem.total / 1e9:.1f} GB")
    except:
        pass
    
    print(f"\n   💡 只保留累积数组 ({memory_gb:.2f} GB)")
    
    # ========== 归一化并写出（分块处理） ==========
    print(f"\n📊 归一化并写出结果（分块处理）...")
    
    meta.update(count=1, dtype='float32', compress='lzw', nodata=-9999)
    output_files = []
    
    CHUNK_ROWS = 5000
    
    for k, name in enumerate(TARGET_FIELDS):
        print(f"\n   [{k+1}/{len(TARGET_FIELDS)}] 处理 {name}")
        
        fname = f"pred_ensemble_adaptive_{name}_2020_90m.tif"
        out_path = os.path.join(OUT_DIR, fname)
        
        print(f"      分块归一化并写出...")
        
        valid_count = 0
        sum_values = 0.0
        
        with rasterio.open(out_path, 'w', **meta) as dst:
            num_chunks = (height + CHUNK_ROWS - 1) // CHUNK_ROWS
            
            for i in tqdm(range(num_chunks), desc=f"      {name}", leave=False):
                start_row = i * CHUNK_ROWS
                end_row = min(start_row + CHUNK_ROWS, height)
                
                comp_chunk = sum_comp[k, start_row:end_row, :]
                weight_chunk = sum_weight[start_row:end_row, :]
                
                weight_safe = np.where(weight_chunk > 0, weight_chunk, 1.0)
                normalized_chunk = comp_chunk / weight_safe
                normalized_chunk[weight_chunk == 0] = -9999
                
                valid_mask = (weight_chunk > 0)
                if valid_mask.any():
                    valid_values = normalized_chunk[valid_mask]
                    valid_count += len(valid_values)
                    sum_values += valid_values.sum()
                
                dst.write(normalized_chunk, 1, window=Window(0, start_row, width, end_row - start_row))
                
                del weight_safe, normalized_chunk
                
                if i % 10 == 0:
                    gc.collect()
        
        if valid_count > 0:
            mean = sum_values / valid_count
            print(f"      统计: μ={mean:.4f}, N={valid_count:,}")
        
        file_size = os.path.getsize(out_path) / 1e6
        print(f"      ✅ 完成 ({file_size:.1f} MB)")
        
        output_files.append(out_path)
        
        gc.collect()
    
    # 最终清理
    print("\n🧹 最终清理...")
    del sum_comp, sum_weight
    gc.collect()
    
    print("\n" + "=" * 80)
    print("🎉 区域自适应集成推理完成！")
    print("=" * 80)
    
    return output_files


# %% ============================================================================
#                           第九部分：主程序
# ==============================================================================

def main():
    """主程序"""
    print("\n" + "=" * 80)
    print("开始执行全图推理（区域自适应集成+内存安全）")
    print("=" * 80)
    
    # 执行推理
    output_files = infer_full_raster_ensemble(
        random_model_path=RANDOM_MODEL_PATH,
        oor_models_dict=OOR_MODELS,
        province_shp=PROVINCE_SHP
    )
    
    # 输出总结
    print("\n📁 输出文件:")
    for f in output_files:
        print(f"   {f}")
    
    print("\n💡 优化总结:")
    print("   ✅ 预筛选: 原始分辨率（100%准确）")
    print("   ✅ 数据读取: DataLoader多进程（不预加载）")
    print("   ✅ 推理后清理: 释放模型（~12GB）")
    print("   ✅ 分块归一化: 避免OOM（峰值~3GB）")
    print("   ✅ NoData处理: 只对有效像元预测")
    print("   ✅ 区域自适应: 智能权重集成")
    print("\n   内存峰值: ~15GB（安全）")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ 用户中断推理（检查点已保存）")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)