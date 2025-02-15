## WenHai

This is the official repository for the paper "Forecasting the Eddying Ocean with a Deep Neural Network".

by Yingzhe Cui, Ruohan Wu, Xiang Zhang, Ziqi Zhu, Bo Liu, Jun Shi, Junshi Chen, Hailong Liu, Shenghui Zhou, Liang Su, Zhao Jing, Hong An, Lixin Wu

### Overview

WenHai can be thought of a hybrid-AI model that utilizes both physical formulae as well as deep learning model. The general workflow is as following: (1) data preparation; (2) air-sea fluxes calculation using bulk formulae; (3) Swin-Transformer-based model inference that generates forecast files in NetCDF format. For convenience, (2) and (3) are implemented in one script.  

### Prerequisite

We provide a packed conda environment `WenHai_env.tgz`. An additional package [aerobulk-python](https://github.com/xgcm/aerobulk-python) is necessary for bulk formulae calculation, please follow the installation instructions therein. 

### Data preparation

#### Initial conditions and atmospheric forcings

1. The fifth generation ECMWF reanalysis ([ERA5](https://doi.org/10.24381/cds.adbb2d47))/ECMWF Integrated Forecasting System ([IFS](https://www.ecmwf.int/en/forecasts/datasets/open-data))

   Eight surface atmospheric variables are used, they are 10-m zonal wind, 10-m meridional wind, mean sea level pressure, 2-m temperature, 2-m dewpoint temperature, surface net short-wave radiation flux, surface downward long-wave radiation flux, and precipitation rate. The hourly (3/6 hourly) reanalysis (forecast) data should be downloaded and processed into daily mean. Then it is remapped to GLORYS 1/12° grid using bilinear interpolation by [ESMF](https://earthsystemmodeling.org/docs/release/ESMF_8_0_1/ESMF_refdoc/node3.html#SECTION03020000000000000000) and [NCO](https://nco.sourceforge.net/nco.html#ncremap). The final input for bulk formulae calculation is a NetCDF file containing the eight variables, each in shape of [Ndays, 2041, 4320], where Ndays is the forecast length. See `sample_ERA5_d0.083.nc` and its attributes for more information. 
   
2. Global Ocean Physics Reanalysis ([GLORYS](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030)) / Global Ocean Physics Analysis and Forecast ([GLO12v4](https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024))

   Five ocean variables are used, they are zonal velocity, meridional velocity, temperature, salinity, and sea surface height. The daily averaged ("P1D-m" in Dataset ID) data should be downloaded and selected according to the this index list along `depth` dimension (23 vertical layers): [1,3,5,7,9,11,13,15,17,19,21,22,23,24,25,26,27,28,29,30,31,32,33] (surface = 1). The initial condition is a NetCDF file containing the five variables, including four 3-D variables each in shape of [1, 23, 2041, 4320] and one 2-D variable in shape of [1, 2041, 4320]. See `sample_GLORYS_23lev.nc` and its attributes for more information.

### Inference

As an example, given 10-day (Jan 01, 2019 ~ Jan 10, 2019) atmospheric variables file `sample_ERA5_d0.083.nc` and 1-day (Jan 01, 2019) ocean initial condition file `sample_GLORYS_23lev.nc`, the following command gives 10-day ocean forecast output in NetCDF format:

```shell
python inference.py --forcing_path 'sample_ERA5_d0.083.nc' --init_path 'sample_GLORYS_23lev.nc' --output_path './init_20190101_10day/'
```

The output directory structure is organized as:

```
init_20190101_10day/
|-- fcst20190102
|   `-- fcst20190102_lead1_byWenHai.nc
|-- fcst20190103
|   `-- fcst20190103_lead2_byWenHai.nc
|-- fcst20190104
|   `-- fcst20190104_lead3_byWenHai.nc
|-- fcst20190105
|   `-- fcst20190105_lead4_byWenHai.nc
|-- fcst20190106
|   `-- fcst20190106_lead5_byWenHai.nc
|-- fcst20190107
|   `-- fcst20190107_lead6_byWenHai.nc
|-- fcst20190108
|   `-- fcst20190108_lead7_byWenHai.nc
|-- fcst20190109
|   `-- fcst20190109_lead8_byWenHai.nc
|-- fcst20190110
|   `-- fcst20190110_lead9_byWenHai.nc
`-- fcst20190111
    `-- fcst20190111_lead10_byWenHai.nc
```

### Sample data

We provide sample initial condition and atmospheric variables in `sample_GLORYS_23lev.nc`, `sample_ERA5_d0.083.nc` and the WenHai model weights `WenHai.onnx`.

The auxiliary data files include `max_GLORYS.npy`, `min_GLORYS.npy`, `max_flux.npy`, `min_flux.npy`, `mask_GLORYS.nc`. 

The above eight files together with `inference.py` as well as packed conda environment `WenHai_env.tgz` are available at this google drive folder.

### Tips

1. The land-ocean mask in GLORYS is different from that in GLO12v4 due to their different production system. We only provide GLORYS land-ocean mask but code modification should be minimal changing it to GLO12v4 mask.
2. Note that tp, strd and ssr provided by ECMWF IFS are accumulated fields but they are instantaneous fields in ERA5. 

### Acknowledgement

We thank the ECMWF for providing ERA5 reanalysis (https://doi.org/10.24381/cds.adbb2d47) and IFS HRES forecasts (https://doi.org/10.21957/open-data). We thank the E.U. Copernicus Marine Service Information for providing GLORYS12 reanalysis (https://doi.org/10.48670/moi-00021) as well as GLO12v4 analysis and forecast (https://doi.org/10.48670/moi-00016). 

#### References

Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C., Dee, D., Thépaut, J-N. (2023): ERA5 hourly data on single levels from 1940 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS), DOI: [10.24381/cds.adbb2d47](https://doi.org/10.24381/cds.adbb2d47) (Accessed on 18-NOV-2024)

Jean-Michel L, Eric G, Romain B-B, Gilles G, Angélique M, Marie D, Clément B, Mathieu H, Olivier LG, Charly R, Tony C, Charles-Emmanuel T, Florent G, Giovanni R, Mounir B, Yann D and Pierre-Yves LT (2021) The Copernicus Global 1/12° Oceanic and Sea Ice GLORYS12 Reanalysis. *Front. Earth Sci.* 9:698876. doi: [10.3389/feart.2021.698876](https://doi.org/10.3389/feart.2021.698876)
