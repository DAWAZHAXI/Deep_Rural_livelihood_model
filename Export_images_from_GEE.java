// ================== 0. 基础资产 ==================
var roi = ee.FeatureCollection("users/dawazhaxi123/boundary/China_albers");

// 农村 100m 栅格
var rural100 = ee.Image("projects/ee-dawa123/assets/GURS_China_100m").select('b1');
// 农村掩膜: 值=2 的为农村
var ruralMask = rural100.eq(2).selfMask();

// 导出统一参数
var exportScale = 90;
var exportCrs   = 'EPSG:3857';

// ================== 1. 构建 2019-2021 6-9 月 Landsat 中值 ==================
var startDate = '2019-06-01';
var endDate   = '2021-09-30';

// 云掩膜 + 缩放
function maskLandsatSR(img) {
  var qa = img.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0) // cloud shadow
    .and(qa.bitwiseAnd(1 << 4).eq(0))    // snow
    .and(qa.bitwiseAnd(1 << 5).eq(0));   // cloud

  // 反射率缩放
  var sr = img.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7'])
              .multiply(0.0000275).add(-0.2);

  // 温度缩放
  var st = img.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('ST_B10')
 

  return img.addBands(sr, null, true)
    .addBands(st, null, true)
    .updateMask(mask);
}

// Landsat 8
var l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
  .filterDate(startDate, endDate)
  .filterBounds(roi)
  .filter(ee.Filter.calendarRange(6, 9, 'month'))
  .map(maskLandsatSR);

// Landsat 9
var l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
  .filterDate(startDate, endDate)
  .filterBounds(roi)
  .filter(ee.Filter.calendarRange(6, 9, 'month'))
  .map(maskLandsatSR);

// 合并
var ls_ic = l8.merge(l9);

// 中值
var ls_med = ls_ic.median().clip(roi);

// 需要的 7 个波段并重命名
var ls_img = ls_med.select(
  ['SR_B4','SR_B3','SR_B2','SR_B5','SR_B6','SR_B7','ST_B10'],
  ['RED','GREEN','BLUE','NIR','SWIR1','SWIR2','TEMP1']
);

// ================== 2. 2019-2021 夜间灯光中值 ==================
var viirs_ic = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
  .filterDate('2019-01-01', '2021-12-31')
  .filterBounds(roi)
  .select('avg_rad');

var viirs_med = viirs_ic.median().clip(roi).rename('VIIRS');

// ================== 3. 重采样到 90m 并套农村掩膜 ==================
var ls90 = ls_img
  .reproject({crs: exportCrs, scale: exportScale})
  .updateMask(ruralMask.reproject({crs: exportCrs, scale: exportScale}));

var viirs90 = viirs_med
  .reproject({crs: exportCrs, scale: exportScale})
  .updateMask(ruralMask.reproject({crs: exportCrs, scale: exportScale}));

// ================== 4. 分开导出到两个文件夹 ==================

// 4.1 Landsat 各波段
var landsatBands = ['RED','GREEN','BLUE','NIR','SWIR1','SWIR2','TEMP1'];
var landsatFolder = 'GEE_exports_2020_Landsat';

landsatBands.forEach(function(bn) {
  var singleBand = ls90.select(bn);
  Export.image.toDrive({
    image: singleBand,
    description: 'Landsat_' + bn + '_2020_90m',
    folder: landsatFolder,
    fileNamePrefix: 'Landsat_' + bn + '_2020_90m',
    region: roi.geometry(),
    scale: exportScale,
    crs: exportCrs,
    maxPixels: 1e13
  });
});

// 4.2 VIIRS 夜间灯光
var viirsFolder = 'GEE_exports_2020_VIIRS';
Export.image.toDrive({
  image: viirs90,
  description: 'VIIRS_2020_90m',
  folder: viirsFolder,
  fileNamePrefix: 'VIIRS_2020_90m',
  region: roi.geometry(),
  scale: exportScale,
  crs: exportCrs,
  maxPixels: 1e13
});

// ================== 5. 可视化（可要可不要） ==================
// 真彩
Map.addLayer(ls90, {bands:['RED','GREEN','BLUE'], min:0, max:0.4}, 'Landsat 90m RGB');
// 夜光
Map.addLayer(viirs90, {min:0, max:40}, 'VIIRS 90m');
// Map.centerObject(roi, 5);
