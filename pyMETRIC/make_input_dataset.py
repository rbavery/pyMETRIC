
import rasterio as rio
import xarray as xar
from affine import Affine
import pyproj
import daymetpy
import pandas as pd
from pathlib import Path
import numpy as np
import pyTSEB.meteo_utils as met
import pyTSEB.resistances as res
from cropmask import io_utils
import os

def get_lat_lon_center(da):
    """
    Gets the center of the raster in lon, lat coordinates by reprojecting to WGS84.
    """
    center = rio.transform.xy(Affine(*da.attrs['transform']), rows=da.sizes['y']//2, cols=da.sizes['x']//2, offset='center')
    outProj =pyproj.Proj(init='epsg:4326')
    inProj = pyproj.Proj(da.attrs['crs'])
    lon,lat = pyproj.transform(inProj,outProj,center[0], center[1])
    return lon, lat
def get_lat_lon_arrs(da):
    """
    Gets the lon, lat coordinates by reprojecting to WGS84, in list of tuple form.
    """
    xs, ys = rio.transform.xy(Affine(*da.attrs['transform']), rows=np.arange(da.sizes['y']), cols=np.arange(da.sizes['x']), offset='center')
    lon_lat_arrs = []
    outProj = pyproj.Proj(init='epsg:4326')
    inProj = pyproj.Proj(da.attrs['crs'])
    lons, lats = pyproj.transform(inProj,outProj,xs, ys)
    return list(zip(lons, lats))

os.chdir("/home/rave/CropMask_RCNN/notebooks")

# test vars
Tr_K = xar.open_rasterio("test_metric/LT05_CU_012006_20020825_20190517_C01_V01_ST/LT05_CU_012006_20020825_20190517_C01_V01_ST.tif", chunks = {'x':500, 'y':500})    .squeeze()
L_dn = xar.open_rasterio("test_metric/LT05_CU_012006_20020825_20190517_C01_V01_ST/LT05_CU_012006_20020825_20190517_C01_V01_DRAD.tif", chunks = {'x':500, 'y':500})    .squeeze()
emis = xar.open_rasterio("test_metric/LT05_CU_012006_20020825_20190517_C01_V01_ST/LT05_CU_012006_20020825_20190517_C01_V01_EMIS.tif", chunks = {'x':500, 'y':500})    .squeeze()

band_paths = list(Path("test_metric/LT05_CU_012006_20020825_20190517_C01_V01_SR/").glob("*B*.tif")) # grab, sort and read in bands as xarr
band_paths = sorted(band_paths)
SR = io_utils.read_bands_lsr(band_paths)
SR = SR.transpose('y', 'x', 'band')

center = get_lat_lon_center(Tr_K)
coords = get_lat_lon_arrs(Tr_K)
coord_arr = np.array(coords, dtype=np.dtype("float,float"))

# need to test if numpy masked arrays or xarray masking works better. using xarr for now
# np.ma.masked_where(L_dn==-9999, L_dn)

emis = emis.where(L_dn!=-9999)
emis = emis * .0001

L_dn = L_dn.where(L_dn!=-9999)
L_dn = L_dn * .001

Tr_K = Tr_K.where(Tr_K!=-9999)
Tr_K = Tr_K * .1

aoi_met = daymetpy.daymet_timeseries(lon=center[0], lat=center[1], start_year=2002, end_year=2002) # used to estimate mock params for aoi

import pvlib
import datetime
from timezonefinder import TimezoneFinder

time = pd.to_datetime(datetime.datetime(2002, 8, 3, 10, 4))
time = pd.DatetimeIndex([time])
tf = TimezoneFinder(in_memory=True)
timezone = tf.certain_timezone_at(lng=center[0], lat=center[1])
time = time.tz_localize(timezone)
solar_df = pvlib.solarposition.get_solarposition(time,center[1], center[0]) # can be made more accurate with temp and pressure from daymet

ltype = "Landsat5"

if ltype in ['Landsat4', 'Landsat5', 'Landsat7']:
    thermal_band = '6'
    # "wb" coefficients from Trezza et al. 2008
    band_sur_dict = {
        '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '7': 6}

    wb_dict = {'1': 0.254, '2': 0.149, '3': 0.147,
                    '4': 0.311, '5': 0.103, '7': 0.036}
elif ltype in ['Landsat8']:

    
    thermal_band = '10'
    band_sur_dict = {
        '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6}
    wb_dict = {'2': 0.254, '3': 0.149, '4': 0.147,
                    '5': 0.311, '6': 0.103, '7': 0.036}
def band_dict_to_array(data_dict, band_dict):
    """
    
    Parameters
    ----------
    data_dict : dict
    band_dict: dict
    Returns
    -------
    ndarray
    """
    return np.array(
        [v for k, v in sorted(data_dict.items())
         if k in band_dict.keys()]).astype(np.float32)

# Convert dictionaries to arrays
wb = band_dict_to_array(wb_dict, band_sur_dict)


def mask_same_shape(function):
    def func(arr1, arr2):
        # tried basing on single nodata vlaue but ARD has negative val artifacts over water
        arr1, arr2= arr1.where(arr2 > 0), arr2.where(arr1 > 0)
        return function(arr1, arr2)
    return func

@mask_same_shape
def savi(nir, r):
    """
    In Landsat 4-7, SAVI = ((Band 4 – Band 3) / (Band 4 + Band 3 + 0.5)) * (1.5).

    In Landsat 8, SAVI = ((Band 5 – Band 4) / (Band 5 + Band 4 + 0.5)) * (1.5).
        
    https://www.usgs.gov/land-resources/nli/landsat/landsat-soil-adjusted-vegetation-index
    """
    
    return ((nir - r) / (nir + r + 0.5)) * (1.5)

@mask_same_shape
def ndvi(nir, r):
    """
    In Landsat 4-7, SAVI = ((Band 4 – Band 3) / (Band 4 + Band 3)).

    In Landsat 8, SAVI = ((Band 5 – Band 4) / (Band 5 + Band 4 )).
    """
    
    return (nir - r) / (nir + r )

def reflectance_to_albedo(refl_sur, wb):
    """Tasumi at-surface albedo
    Parameters
    ----------
    refl_sur : ndarray
    wb :
    Returns
    -------
    ndarray
    References
    ----------
    .. [1] Tasumi, M., Allen, R., and Trezza, R. (2008). At-surface reflectance
       and albedo from satellite for operational calculation of land surface
       energy balance. Journal of Hydrologic Engineering 13(2):51-63.
       https://doi.org/10.1061/(ASCE)1084-0699(2008)13:2(51)
    """
    return np.sum(refl_sur * wb, axis=2)

def savi_lai_func(savi):
    """Compute leaf area index (LAI) from SAVI
    Parameters
    ----------
    savi : array_like
        Soil adjusted vegetation index.
        
    Returns
    -------
    ndarray
    References
    ----------
    .. [1] Allen, R., Tasumi, M., & Trezza, R. (2007). Satellite-Based Energy
       Balance for Mapping Evapotranspiration with Internalized Calibration
       (METRIC)-Model. Journal of Irrigation and Drainage Engineering, 133(4).
       https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)
    """
    return np.clip((11. * np.power(savi, 3)), 0, 6)


def ndvi_lai_func(ndvi):
    """Compute leaf area index (LAI) from NDVI
    Parameters
    ----------
    ndvi : array_like
        Normalized difference vegetation index.
    Returns
    -------
    ndarray
    References
    ----------
    .. [1] Trezza and Allen 2014?
    """
    return np.clip((7. * np.power(ndvi, 3)), 0, 6)

# from nlcd metric file on DRI pymetric
zom_nlcd_remap = {
  "11": "0.0005", # open water
  "12": "0.005", # perennial ice/snow
  "21": "0.05", # developed, open space
  "22": "0.08", # developed, low intensity
  "23": "0.1", # developed, medium
  "24": "0.2", # developed, high
  "31": "0.1", # barren land (rock sand clay)
  "32": "0.005",
  "41": "perrier", # deciduous forest
  "42": "perrier", # evergreen forest
  "43": "perrier", # mixed forest
  "51": "0.2", # dwarf scrub
  "52": "0.2", # shrb/scrub
  "71": "0.05", # grassland/herbaceaous
  "72": "0.03", # sedge/Herbaceaous
  "81": "LAI",
  "82": "LAI", # pasture/hay
  "90": "0.4", # cultivated crops
  "94": "0.2", # woody wetlands
  "95": "0.01" # emergent herbaceaous wetlands
 }
# https://www.mrlc.gov/data/legends/national-land-cover-database-2011-nlcd2011-legend

# # METRIC model

# mocks
alt=0
T_A_K = 25.0+273.15
u = 2 #m/s
ea = 20 # mb, mock value under the saturation vapor pressure at 25 Celsius
p = 1013 # mb
S_dn = 500 # flux density, guestimate based on center of image and rough day of year daymet data
L_dn = L_dn
emis = emis
z_0M = .018 # surface roughness length for momentum transport, see https://reader.elsevier.com/reader/sd/pii/S0022169498002534?token=B4ADFCE769E6A06E951B6DDF5DBBF54EB29B4C76713AA74F1322C942C1B9C560F655D77CD6EDA4CA9D2E71C711AD7167
h_C = 2 # mock canopy height of 2 meters
d_0 = res.calc_d_0(xar.ones_like(Tr_K)*h_C)
z_u = 2 # height of windspeed measurement
z_T = 2 # height of air temperature measurement
LE_hot=0
use_METRIC_resistance=True
calcG_params=[[0], 0.15]
UseL=False
UseDEM=False

# end member search

from pyMETRIC import endmember_search

VI = ndvi(SR.sel(band=4), SR.sel(band=3))

VI_MAX = 0.95

albedo = reflectance_to_albedo(SR.where(SR>0), wb)

# Reduce potential ET based on vegetation density based on Allen et al. 2013
ET_r_f_cold = xar.ones_like(Tr_K) * 1.05
ET_bare_soil = xar.zeros_like(Tr_K)
ET_r_f_cold = xar.where(VI < VI_MAX, 1.05/VI_MAX * VI, ET_r_f_cold) # Eq. 4 [Allen 2013]

ET_r_f_hot = VI * ET_r_f_cold + (1.0 - VI) * ET_bare_soil # Eq. 5 [Allen 2013]

# Compute normalized temperatures by adiabatic correction
gamma_w = met.calc_lapse_rate_moist(Tr_K,
                                    ea,
                                    p)

Tr_datum = Tr_K + gamma_w * alt
Ta_datum = T_A_K + gamma_w * alt

cv_ndvi, _, _ = endmember_search.moving_cv_filter(VI, (11, 11))
cv_lst, _, std_lst = endmember_search.moving_cv_filter(Tr_datum, (11, 11))
cv_albedo,_, _ = endmember_search.moving_cv_filter(albedo, (11, 11))

cold_pixel, hot_pixel = endmember_search.esa(VI,
                                Tr_datum,
                                cv_ndvi,
                                std_lst,
                                cv_albedo)

from pyMETRIC.METRIC import pet_asce

LE_potential = pet_asce(Ta_datum,
                              u,
                              ea,
                              p,
                              S_dn,
                              z_u,
                              z_T,
                              f_cd=1,
                              reference=True)


from pyMETRIC.METRIC import METRIC

        
flag, R_nl1, LE1, H1, G1, R_A1, u_friction, L, n_iterations = METRIC(Tr_K,
                T_A_K,
                u,
                ea,
                p,
                S_dn,
                L_dn,
                emis,
                z_0M,
                d_0,
                z_u,
                z_T,
                cold_pixel,
                hot_pixel,
                LE_potential,
                LE_hot=0,
                use_METRIC_resistance = use_METRIC_resistance,
                calcG_params=calcG_params,
                UseDEM=UseDEM)
