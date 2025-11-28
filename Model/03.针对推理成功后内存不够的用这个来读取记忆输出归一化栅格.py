"""
快速归一化脚本 - 从检查点恢复
"""
import os
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

print("从检查点加载并归一化...")

# 配置
CHECKPOINT_PATH = r"F:\model_outputs_2020_resnet_optimized\maps_2020_optimized\checkpoint.npz"
OUT_DIR = r"F:\model_outputs_2020_resnet_optimized\maps_2020_optimized"
DAY_TIF = r"F:\Landsat_NL_Mector_90m_zscore\Landsat_RED_2020_90m_zscore.tif"
TARGET_FIELDS = ["F", "F_NF", "NF_F", "NF"]
CHUNK_ROWS = 5000

# 获取影像信息
with rasterio.open(DAY_TIF) as src:
    height = src.height
    width = src.width
    meta = src.meta.copy()

meta.update(count=1, dtype='float32', compress='lzw', nodata=-9999)

# 加载检查点
print("⏳ 加载检查点（需要1-2分钟）...")
data = np.load(CHECKPOINT_PATH)
sum_comp = data['sum_comp']
sum_weight = data['sum_weight']

print(f"✅ 检查点加载成功: {sum_comp.shape}")

# 归一化并写出
for k, name in enumerate(TARGET_FIELDS):
    print(f"\n[{k+1}/4] 处理 {name}...")
    
    fname = f"pred_optimized_{name}_2020_90m.tif"
    out_path = os.path.join(OUT_DIR, fname)
    
    valid_count = 0
    sum_values = 0.0
    sum_squares = 0.0
    
    with rasterio.open(out_path, 'w', **meta) as dst:
        num_chunks = (height + CHUNK_ROWS - 1) // CHUNK_ROWS
        
        for i in tqdm(range(num_chunks), desc=f"   写出{name}"):
            start_row = i * CHUNK_ROWS
            end_row = min(start_row + CHUNK_ROWS, height)
            
            comp_chunk = sum_comp[k, start_row:end_row, :]
            weight_chunk = sum_weight[start_row:end_row, :]
            
            weight_safe = np.where(weight_chunk > 0, weight_chunk, 1.0)
            normalized_chunk = comp_chunk / weight_safe
            normalized_chunk[weight_chunk == 0] = -9999
            
            # 统计
            valid_mask = (weight_chunk > 0)
            if valid_mask.any():
                valid_values = normalized_chunk[valid_mask]
                valid_count += len(valid_values)
                sum_values += valid_values.sum()
                sum_squares += (valid_values ** 2).sum()
            
            dst.write(normalized_chunk, 1, window=Window(0, start_row, width, end_row - start_row))
            
            del weight_safe, normalized_chunk
    
    # 统计
    if valid_count > 0:
        mean = sum_values / valid_count
        variance = (sum_squares / valid_count) - (mean ** 2)
        std = np.sqrt(max(variance, 0))
        print(f"   统计: μ={mean:.4f}, σ={std:.4f}, N={valid_count:,}")
    
    file_size = os.path.getsize(out_path) / 1e6
    print(f"   ✅ 完成 ({file_size:.1f} MB)")

print("\n🎉 归一化完成！")