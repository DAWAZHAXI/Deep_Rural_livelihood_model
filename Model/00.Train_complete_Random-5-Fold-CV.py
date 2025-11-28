#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# %% [markdown]第一部分：完整训练脚本
# # 完整训练脚本（内存缓存优化版）
# ## 优化：数据预加载到内存 + 多线程 + 进度条

# %% 导入所有库
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import rasterio
import geopandas as gpd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

print("✅ 库导入成功")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

# %% [markdown]
# ## 第一部分：内存缓存版数据集

# %% 定义 PatchPointDataset（内存缓存版）
# %% [markdown]
# ## 第一部分：内存缓存版数据集

# %% 定义 PatchPointDataset（内存缓存版）
class PatchPointDataset(Dataset):
    """点样本数据集（内存缓存版 - 解决pickling问题）"""

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

        # 如果数据已缓存，关闭文件句柄
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

            # 读取日间数据
            day_stack = []
            for src in self.day_srcs:
                arr = src.read(1, window=window, boundless=True, masked=True)
                if arr.mask.all() or np.isnan(arr.filled(0)).all():
                    return None
                day_stack.append(arr.filled(0).astype(np.float32))

            day_arr = np.stack(day_stack, axis=0)

            # 读取夜光数据
            night_arr = self.night_src.read(1, window=window, boundless=True, masked=True)
            if night_arr.mask.all() or np.isnan(night_arr.filled(0)).all():
                return None
            night_arr = night_arr.filled(0).astype(np.float32)[np.newaxis, :, :]

            # 标签检查
            vals = [row[f] for f in self.target_fields]
            if any((v is None) or (isinstance(v, float) and np.isnan(v)) for v in vals):
                return None

            # 处理标签
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

        # 计算内存占用
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
        """切换模式：'train' 开启增强，'val' / 'test' 关闭增强"""
        assert mode in ["train", "val", "test"], f"Unsupported mode: {mode}"
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

        # 从内存缓存读取
        if self.cached_patches is not None:
            data = self.cached_patches[real_idx]
            day = data['day'].copy()
            night = data['night'].copy()
            y = data['y'].copy()
        else:
            # 从磁盘读取（如果没缓存）
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

        # 数据增强（只在训练模式）
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

print("✅ PatchPointDataset（内存缓存版）定义完成")




# In[ ]:


# %% [markdown]第二部分：模型定义
# ## 第二部分：模型定义

# %% ResNet模块
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

print("✅ 模型定义完成")

# %% 损失函数
def dirichlet_nll(y_true, alpha, eps=1e-7):
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
    mask = np.isfinite(y_true).all(axis=1) & np.isfinite(y_pred).all(axis=1)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.shape[0] < 2:
        return np.nan, [np.nan] * y_true.shape[1]

    r2_each = [r2_score(y_true[:, i], y_pred[:, i]) 
               for i in range(y_true.shape[1])]
    r2_mean = float(np.mean(r2_each))
    return r2_mean, r2_each

print("✅ 损失函数定义完成")





# In[ ]:


# %% [markdown]第三部分：配置参数
# ## 第三部分：配置参数

# %% 配置
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
TARGET_FIELDS = ["F", "F_NF", "NF_F", "NF"]

OUT_DIR = r"F:\model_outputs_2020_resnet_optimized"
os.makedirs(OUT_DIR, exist_ok=True)

# 快速测试模式开关
QUICK_TEST = False  # 改为 False 进行完整训练   True

if QUICK_TEST:
    print("\n" + "="*70)
    print("⚡ 快速测试模式")
    print("="*70)
    MAX_SAMPLES = 500
    PATCH_SIZE = 64
    BATCH_SIZE = 32
    EPOCHS = 20
    PATIENCE = 5
    BASE_LR_LIST = [1e-3]
    WD_LIST = [1e-2]
    FRACTIONS = [0.1, 1.0]
    N_REPEATS = 1
else:
    print("\n" + "="*70)
    print("🚀 完整训练模式")
    print("="*70)
    MAX_SAMPLES = None
    PATCH_SIZE = 64
    BATCH_SIZE = 256   #64
    EPOCHS = 300
    PATIENCE = 15
    BASE_LR_LIST = [1e-3, 1e-4, 1e-5] # [1e-2, 1e-3, 1e-4, 1e-5]
    WD_LIST = [1e-1, 1e-2, 1e-4]  # [1e0, 1e-1, 1e-2, 1e-3]
    FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.00]
    N_REPEATS = 1

# ⭐⭐ 新增：阶段1专用“轻量版”设置（推荐）
EPOCHS_STAGE1 = 60          # 网格搜索时最多 60 轮
PATIENCE_STAGE1 = 8         # 8 轮不提升就早停
FRACTION_STAGE1 = 0.3       # 只用训练集的 30% 进行超参搜索

# ==========================================================# ==========================================================
# 额外控制开关：是否运行阶段1/阶段2 & 是否断点续跑
# ==========================================================# ==========================================================
RUN_STAGE1 = False   # 是否执行阶段1（网格搜索），现在你可以先关掉# ========================================================
RUN_STAGE2 = True    # 是否执行阶段2（数据量实验）            # ==========================================================

RESUME_STAGE1 = True  # True = 如果已有 CSV，按其中进度续跑   # ==========================================================
RESUME_STAGE2 = True  # True = 如果已有 CSV，按其中进度续跑   # ==========================================================

# 是否使用手动指定的超参数（推荐先 True，等你有完整的阶段1结果再切 False）# =================================================
USE_MANUAL_BEST_HPARAMS = True                             # ==========================================================
MANUAL_BEST_LR = 3e-4                                      # ==========================================================
MANUAL_BEST_WD = 1e-3                                        # =========================================================
# ==========================================================# ==========================================================
# ==========================================================# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0 #min(4, os.cpu_count() or 2)  # 恢复多线程

print(f"✅ 配置完成")
print(f"   设备: {DEVICE}")
print(f"   DataLoader workers: {NUM_WORKERS}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")




# In[ ]:


# %% [markdown]第四部分：训练函数
# ## 第四部分：训练函数

# %% 训练函数
def train_one_setting(base_ds, train_idx, val_idx, test_idx,
                      lr, weight_decay, fraction, repeat_id, fold_id,
                      epochs=None, patience=None):
    """单次训练和评估（完全基于 base_ds 内存缓存）"""
        # 如果没传，就用全局“正式版”参数
    if epochs is None:
        epochs = EPOCHS
    if patience is None:
        patience = PATIENCE
    # 从 train_idx 中随机抽取 fraction 比例
    n_train = int(len(train_idx) * fraction)
    n_train = max(n_train, 1)
    rng = np.random.RandomState(seed=fold_id * 100 + repeat_id * 7 + int(fraction * 100))
    sub_train_idx = rng.choice(train_idx, size=n_train, replace=False)

    # 用同一个 base_ds，创建不同的 Subset
    train_subset = Subset(base_ds, list(sub_train_idx))
    val_subset   = Subset(base_ds, list(val_idx))
    test_subset  = Subset(base_ds, list(test_idx))

    # DataLoader（多线程）
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
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-7
    )

    best_val_r2 = -1e9
    best_state = None
    epochs_no_improve = 0

    pbar = tqdm(range(1, epochs + 1), desc=f"Fold{fold_id} frac={fraction:.2f}")

    for epoch in pbar:
        # ========= 训练 =========
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

        # ========= 验证 =========
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

        # 早停时用 patience
        if val_r2_mean > best_val_r2 + 1e-4:
            best_val_r2 = val_r2_mean
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                pbar.set_description(f"Fold{fold_id} frac={fraction:.2f} [早停]")
                break

    # ========= 测试 =========
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

    # 保存模型（只在 fraction=1.0 时）
    model_path = None
    if abs(fraction - 1.0) < 1e-6:
        model_path = os.path.join(
            OUT_DIR,
            f"model_fold{fold_id}_rep{repeat_id}_lr{lr:g}_wd{weight_decay:g}.pth",
        )
        torch.save(best_state, model_path)

    metrics = {
        "fold": fold_id, "fraction": fraction, "repeat": repeat_id,
        "lr": lr, "weight_decay": weight_decay,
        "n_train": len(sub_train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        "val_r2_best": best_val_r2, "test_r2_mean": test_r2_mean,
    }
    for name, r2v in zip(TARGET_FIELDS, test_r2_each):
        metrics[f"test_r2_{name}"] = r2v

    if model_path is not None:
        metrics["model_path"] = model_path

    return metrics

print("✅ 训练函数定义完成")



# In[ ]:


# %% [markdown]第五部分：构建数据集
# ## 第五部分：构建数据集

# %% 构建数据集
print("\n" + "="*70)
print("📊 构建基础数据集...")
print("="*70)

base_ds = PatchPointDataset(
    LABEL_SHP, DAY_TIFS, NIGHT_TIF, TARGET_FIELDS,
    patch_size=PATCH_SIZE, mode='val', check_valid=True, max_samples=MAX_SAMPLES
)

n_total = len(base_ds)
all_idx = np.arange(n_total)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds = list(kf.split(all_idx))

print(f"✅ 数据集准备完成，共{n_total}个有效样本，划分为5折")





# In[ ]:


# # %% [markdown]第六部分：阶段1 - 网格搜索（可选 + 断点续跑）
# # ## 第六部分：阶段1 - 网格搜索（可选 + 断点续跑）

# GRID_CSV = os.path.join(OUT_DIR, "stage1_grid_search.csv")

# if RUN_STAGE1:
#     print("\n" + "="*70)
#     print("阶段1：网格搜索最佳超参数（支持断点续跑）")
#     print("="*70)

#     # 1. 如果选择续跑且已有结果文件，就读进来
#     if RESUME_STAGE1 and os.path.exists(GRID_CSV):
#         df_grid = pd.read_csv(GRID_CSV)
#         print(f"🔁 检测到已有阶段1结果，将在其基础上续跑：{len(df_grid)} 条记录")
#     else:
#         df_grid = pd.DataFrame()

#     # 2. 已完成的 (lr, wd, fold) 组合
#     done_keys = set()
#     if not df_grid.empty and "fold" in df_grid.columns:
#         for _, row in df_grid.iterrows():
#             try:
#                 done_keys.add((float(row["lr"]), float(row["weight_decay"]), int(row["fold"])))
#             except Exception:
#                 continue

#     new_results = []

#     # 3. 正式网格搜索
#     for lr in BASE_LR_LIST:
#         for wd in WD_LIST:
#             print(f"\n--- lr={lr:g}, wd={wd:g} ---")

#             for fold_id, (trainval_idx, test_idx) in enumerate(folds, start=1):
#                 key = (float(lr), float(wd), int(fold_id))
#                 if key in done_keys:
#                     print(f"⏭️ 跳过已完成：lr={lr:g}, wd={wd:g}, fold={fold_id}")
#                     continue

#                 rng = np.random.RandomState(seed=fold_id)
#                 rng.shuffle(trainval_idx)
#                 n_val = max(1, len(trainval_idx) // 4)
#                 val_idx = trainval_idx[:n_val]
#                 train_idx = trainval_idx[n_val:]

#                 try:
#                     m = train_one_setting(
#                         base_ds, train_idx, val_idx, test_idx,
#                         lr, wd, 1.0, 0, fold_id,
#                     )
#                     # 确保 lr / wd 写入结果（train_one_setting 里没写就补一下）
#                     m["lr"] = lr
#                     m["weight_decay"] = wd
#                     new_results.append(m)

#                     # 每次更新后立刻写盘，防止停电丢进度
#                     df_all = pd.concat([df_grid, pd.DataFrame(new_results)], ignore_index=True)
#                     df_all.to_csv(GRID_CSV, index=False)

#                 except Exception as e:
#                     print(f"   ❌ Fold{fold_id} 出错: {e}")
#                     continue

#     # 4. 最终合并保存
#     if new_results:
#         df_grid = pd.concat([df_grid, pd.DataFrame(new_results)], ignore_index=True)
#         df_grid.to_csv(GRID_CSV, index=False)
#         print(f"\n✅ 阶段1完成，结果保存到: {GRID_CSV}")
#     else:
#         print("\nℹ️ 阶段1没有新增结果（可能全部已完成）")
# else:
#     print("\n[阶段1] 已跳过网格搜索（RUN_STAGE1 = False）")


# In[ ]:


# 在阶段2开始前，确定要使用的超参数 best_lr / best_wd
# ==========================================================
if USE_MANUAL_BEST_HPARAMS:
    best_lr = MANUAL_BEST_LR
    best_wd = MANUAL_BEST_WD
    print(f"\n⚠️ 阶段2使用手动指定超参数: lr={best_lr:g}, wd={best_wd:g}")
else:
    # 尝试从阶段1结果中自动选出最优 (lr, wd)
    GRID_CSV = os.path.join(OUT_DIR, "stage1_grid_search.csv")
    if os.path.exists(GRID_CSV):
        df_grid = pd.read_csv(GRID_CSV)
        if "error" in df_grid.columns:
            df_ok = df_grid[df_grid["error"].isna()]
        else:
            df_ok = df_grid

        if not df_ok.empty:
            best_config = df_ok.groupby(["lr", "weight_decay"])["test_r2_mean"].mean().idxmax()
            best_lr, best_wd = best_config
            print(f"\n🏆 从阶段1结果中读取超参数: lr={best_lr:g}, wd={best_wd:g}")
        else:
            best_lr, best_wd = MANUAL_BEST_LR, MANUAL_BEST_WD
            print(f"\n⚠️ 阶段1结果为空，回退到手动超参数: lr={best_lr:g}, wd={best_wd:g}")
    else:
        best_lr, best_wd = MANUAL_BEST_LR, MANUAL_BEST_WD
        print(f"\n⚠️ 未找到阶段1结果文件，回退到手动超参数: lr={best_lr:g}, wd={best_wd:g}")


# In[ ]:


# %% [markdown]第七部分：阶段2 - 数据量测试（支持断点续跑）
# ## 第七部分：阶段2 - 数据量测试（支持断点续跑）

FINAL_CSV = os.path.join(OUT_DIR, "stage2_fraction_results.csv")

print("\n" + "="*70)
print(f"阶段2：测试数据量影响（lr={best_lr:g}, wd={best_wd:g}，支持断点续跑）")
print("="*70)

# 1. 如果已经有阶段2结果（上次跑了一部分），就读进来
if os.path.exists(FINAL_CSV):
    df_final = pd.read_csv(FINAL_CSV)
    print(f"🔁 检测到已有阶段2结果，将在其基础上续跑：{len(df_final)} 条记录")
else:
    df_final = pd.DataFrame()

# 2. 把已经完成的 (fraction, repeat, fold) 组合记下来，避免重复跑
done_keys = set()
if not df_final.empty:
    if "error" in df_final.columns:
        df_ok = df_final[df_final["error"].isna()]
    else:
        df_ok = df_final

    for _, row in df_ok.iterrows():
        try:
            frac_done = float(row["fraction"])
            rep_done = int(row["repeat"])
            fold_done = int(row["fold"])
            done_keys.add((frac_done, rep_done, fold_done))
        except Exception:
            continue

all_results = df_final.to_dict("records") if not df_final.empty else []

# 3. 正式跑 FRACTIONS × N_REPEATS × 5-fold
for frac in FRACTIONS:
    print(f"\n>>> Fraction = {frac*100:.0f}%")
    for rep in range(N_REPEATS):
        for fold_id, (trainval_idx, test_idx) in enumerate(folds, start=1):
            key = (float(frac), int(rep), int(fold_id))
            if key in done_keys:
                print(f"⏭️ 跳过已完成：frac={frac:.2f}, rep={rep}, fold={fold_id}")
                continue

            rng = np.random.RandomState(seed=fold_id)
            rng.shuffle(trainval_idx)
            n_val = max(1, len(trainval_idx) // 4)
            val_idx = trainval_idx[:n_val]
            train_idx = trainval_idx[n_val:]

            try:
                m = train_one_setting(
                    base_ds, train_idx, val_idx, test_idx,
                    best_lr, best_wd, frac, rep, fold_id,
                )
                all_results.append(m)

                # 🔥 每完成一个组合就立刻写盘，防止停电重来
                pd.DataFrame(all_results).to_csv(FINAL_CSV, index=False)

            except Exception as e:
                print(f"   ❌ Fold{fold_id}/rep{rep}/frac{frac:.2f} 出错: {e}")
                all_results.append({
                    "fold": fold_id,
                    "fraction": frac,
                    "repeat": rep,
                    "lr": best_lr,
                    "weight_decay": best_wd,
                    "error": str(e),
                })
                pd.DataFrame(all_results).to_csv(FINAL_CSV, index=False)
                continue

# 4. 最终保存一次，防止万一
df_final = pd.DataFrame(all_results)
df_final.to_csv(FINAL_CSV, index=False)

print("\n" + "="*70)
print("✅ 阶段2训练完成！")
print(f"   阶段2结果: {FINAL_CSV}")
print("="*70)

# 清理
base_ds.close()



# In[ ]:


# %% [markdown]第八部分：快速总结（兼容“阶段1已注释”的情况）
# ## 第八部分：快速总结（兼容“阶段1已注释”的情况）

print("\n" + "="*70)
print("📊 训练总结")
print("="*70)

# 尝试读取阶段1结果（如果你以后恢复阶段1，仍然能用）
GRID_CSV = os.path.join(OUT_DIR, "stage1_grid_search.csv")
if os.path.exists(GRID_CSV):
    df_grid = pd.read_csv(GRID_CSV)
    if not df_grid.empty:
        if "error" in df_grid.columns:
            df_grid_ok = df_grid[df_grid["error"].isna()]
        else:
            df_grid_ok = df_grid

        if not df_grid_ok.empty:
            print("\n【阶段1：网格搜索】")
            print(f"  测试超参组合: {len(df_grid_ok.groupby(['lr', 'weight_decay']))}")
            print(f"  总实验次数: {len(df_grid_ok)}")
            # 这里不再强行打印 best_r2（因为你可能用了手动超参）
            best_row = df_grid_ok.loc[df_grid_ok['test_r2_mean'].idxmax()]
            print(f"  最佳超参: lr={best_row['lr']:.0e}, wd={best_row['weight_decay']:.0e}")
            print(f"  对应测试R²: {best_row['test_r2_mean']:.4f}")
else:
    print("\n【阶段1：网格搜索】")
    print("  已跳过或尚未运行（未找到 stage1_grid_search.csv）")

# 阶段2汇总
if os.path.exists(FINAL_CSV):
    df_final = pd.read_csv(FINAL_CSV)
    if not df_final.empty:
        df_clean = df_final[df_final['error'].isna()] if 'error' in df_final.columns else df_final
        if not df_clean.empty:
            print("\n【阶段2：数据量测试】")
            print(f"  测试比例: {sorted(df_clean['fraction'].unique())}")
            print(f"  总实验次数: {len(df_clean)}")
            print("\n  各数据比例性能:")
            for frac in sorted(df_clean['fraction'].unique()):
                subset = df_clean[df_clean['fraction'] == frac]
                mean_r2 = subset['test_r2_mean'].mean()
                std_r2 = subset['test_r2_mean'].std()
                print(f"    {frac*100:5.1f}%: R²={mean_r2:.4f}±{std_r2:.4f}")

print("\n" + "="*70)

