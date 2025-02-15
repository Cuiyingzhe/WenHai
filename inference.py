from datetime import datetime,timedelta
import os
import multiprocessing
import numpy as np
import xarray as xr
from metpy.calc import specific_humidity_from_dewpoint
from metpy.units import units
from aerobulk.flux import noskin_np
import pandas as pd
import argparse
import onnxruntime as ort

# pass in arguments

parser = argparse.ArgumentParser()
parser.add_argument('--forcing_path', type = str)
parser.add_argument('--init_path', type = str)
parser.add_argument('--output_path', type = str)
args = parser.parse_args()
forcing_path = args.forcing_path
init_path = args.init_path
output_path = args.output_path

# bulk formulae functions 

def calculate_noskin_np(args):
    t0, t2m, h2m, u, v, msl = args
    return noskin_np(t0, t2m, h2m, u, v, msl, 'ncar', 2, 10, 4, False)

def calc_bulk_flux(u0,v0,t0,forcing_path,i):
    # read in surface atmospheric and ocean variables
    ds = xr.open_dataset(forcing_path).isel(time=i)
    t0 = ((max_GLORYS[0,46] - min_GLORYS[0,46])*t0.astype(np.float32) + min_GLORYS[0,46])*mask[0,46]+273.15
    t0 *= mask[0,46]
    u0 = ((max_GLORYS[0,0] - min_GLORYS[0,0])*u0.astype(np.float32)+min_GLORYS[0,0])*mask[0,0]
    v0 = ((max_GLORYS[0,23] - min_GLORYS[0,23])*v0.astype(np.float32)+min_GLORYS[0,23])*mask[0,23]
    t2m, d2m, u10, v10, mtpr, ssr, strd, msl = [da.values for da in [ds.t2m, ds.d2m, ds.u10, ds.v10, ds.mtpr, ds.ssr, ds.strd, ds.msl]]
    ds.close()
    ssr /= 3600 # J m-2 to W m-2, hourly
    strd /= 3600
    mtpr /= 1000 # mm/s (kg/m^2/s) to m/s
    h2m = specific_humidity_from_dewpoint(msl * units.Pa, d2m * units.K).to('kg/kg').magnitude
    h2m = np.nan_to_num(h2m)

    # bulk formulae, the speed can be further optimized
    pool = multiprocessing.Pool(processes=12)
    results = pool.map(calculate_noskin_np, [[t0[:,i], t2m[:,i], h2m[:,i], u10[:,i]-u0[:,i], v10[:,i]-v0[:,i], msl[:,i]] for i in range(4320)])
    pool.close()
    qe, qh, taux, tauy, evap = zip(*results)
    qe = np.array(qe).T.reshape(t0.shape)
    qh = np.array(qh).T.reshape(t0.shape)
    taux = np.array(taux).T.reshape(t0.shape)
    tauy = np.array(tauy).T.reshape(t0.shape)
    evap = np.array(evap).T.reshape(t0.shape)/1000

    # net surface thermal (longwave) radiation
    sigma = 5.67e-8 #W/m^2 per K^4 
    ql = strd-sigma*(t0**4)
    
    # normalization
    bulk_flux = np.stack((ql,ssr,qh,qe,taux,tauy,evap,mtpr), axis = 0)
    bulk_flux = np.nan_to_num(bulk_flux)
    bulk_flux = (bulk_flux - min_flux)/(max_flux-min_flux)
    bulk_flux *= mask[:,0]
    return bulk_flux.astype(np.float16).clip(0,1)[None]

# set up onnxruntime
providers = ['CUDAExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
session = ort.InferenceSession('WenHai.onnx', providers=providers)
input_info = session.get_inputs()
name1 = input_info[0].name
name2 = input_info[1].name

# read data
ds = xr.open_dataset(init_path)
longitude = ds.longitude.values
latitude = ds.latitude.values
depth = ds.depth.values
init_date = datetime.fromtimestamp((ds.time[0].values - np.datetime64(0, "s")) / np.timedelta64(1, "s"))
init = np.concatenate([ds.uo.values, ds.vo.values, ds.thetao.values, ds.so.values, ds.zos.values[None]], axis = 1) 
ds.close()
min_GLORYS = np.load('min_GLORYS.npy').reshape(1,-1,1,1)
max_GLORYS = np.load('max_GLORYS.npy').reshape(1,-1,1,1)
min_flux = np.load('min_flux.npy').reshape(-1,1,1)
max_flux = np.load('max_flux.npy').reshape(-1,1,1)
mask = xr.open_dataset('mask_GLORYS.nc').mask.values[None]

# initial condition
init -= min_GLORYS
init /= (max_GLORYS-min_GLORYS)
init = np.nan_to_num(init)

# set forecast length to number of days in provided forcing file
nday = len(xr.open_dataset(forcing_path).time)

for i in range(nday):
    fcst_date = (init_date + timedelta(days = 1+i)).strftime('%Y%m%d')

    # set autoregressive initial condition 
    if i != 0:
        input_tensor = last_step_output
    else:
        input_tensor = init 
    
    # send u0, v0, t0 and atmospheric file to calc_bulk_flux
    bulk_flux = calc_bulk_flux(input_tensor[0,0],input_tensor[0,23],input_tensor[0,46], forcing_path, i)
    # inference
    inputs = {name1: input_tensor.astype(np.float16).clip(0,1), name2:bulk_flux}  
    output = session.run(None, inputs)[0]
    output += input_tensor
    output *= mask
    output = output.clip(0,1)
    last_step_output = output.copy()

    # write forecasts in NetCDF format
    output = output * (max_GLORYS - min_GLORYS) + min_GLORYS
    output[mask==0]=np.nan
    output = output[0].astype(np.float32)
    t = output[46:69]
    s = output[69:92]
    u = output[:23]
    v = output[23:46]
    ssh = output[-1]
    da_s = xr.DataArray(
        data=s,
        dims = ["depth", "latitude", "longitude"],
        coords=dict(
            depth = (["depth"], depth),
            longitude=(["longitude"], longitude),
            latitude=(["latitude"], latitude),
        ),
        attrs=dict(
            longitudeg_name="Salinity",
            units="1e-3",
        ),
    ).expand_dims(time = [pd.to_datetime(fcst_date)])
    da_t = xr.DataArray(
        data=t,
        dims = ["depth", "latitude", "longitude"],
        coords=dict(
            depth = (["depth"], depth),
            longitude=(["longitude"], longitude),
            latitude=(["latitude"], latitude),
        ),
        attrs=dict(
            longitudeg_name="Temperature",
            units="degrees_C",
        ),
    ).expand_dims(time = [pd.to_datetime(fcst_date)])
    da_u = xr.DataArray(
            data=u,
                dims = ["depth", "latitude", "longitude"],
                coords=dict(
                    depth = (["depth"], depth),
                    longitude=(["longitude"], longitude),
                    latitude=(["latitude"], latitude),
                ),
                attrs=dict(
                    longitudeg_name="Eastward velocity",
                    units="m s-1",
                ),
            ).expand_dims(time = [pd.to_datetime(fcst_date)])
    da_v = xr.DataArray(
        data=v,
        dims = ["depth", "latitude", "longitude"],
        coords=dict(
            depth = (["depth"], depth),
            longitude=(["longitude"], longitude),
            latitude=(["latitude"], latitude),
        ),
        attrs=dict(
            longitudeg_name="Northward velocity",
            units="m s-1",
        ),
    ).expand_dims(time = [pd.to_datetime(fcst_date)])
    da_ssh = xr.DataArray(
            data=ssh,
            dims = ["latitude", "longitude"],
            coords=dict(
                longitude=(["longitude"], longitude),
                latitude=(["latitude"], latitude),
            ),
            attrs=dict(
                longitudeg_name="Sea surface height",
                units="m",
            ),
    ).expand_dims(time = [pd.to_datetime(fcst_date)])
    ds = xr.Dataset({'thetao': da_t, 'so':da_s,'uo':da_u,'vo':da_v,'zos':da_ssh})
    if not os.path.exists(f"{output_path}/fcst{fcst_date}/"):
        os.makedirs(f"{output_path}/fcst{fcst_date}/")
    ds.to_netcdf(f"{output_path}/fcst{fcst_date}/fcst{fcst_date}_lead{i+1}_byWenHai.nc",unlimited_dims=["time"])
    ds.close()
