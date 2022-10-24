#------------------------------------------------------------------------------
# Install relevant packages
#------------------------------------------------------------------------------
# Load relevant packages
import datetime as dtmod
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os.path
import re
import netCDF4 as nc
import sys
import scipy.stats
import pandas as pd
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import matplotlib
import matplotlib.colors as mcolors
import urllib.request
import requests
from bs4 import BeautifulSoup
import get_file_path
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

#20221013 20221001 20221231 0.33 0.34 0.33 region 33.5 42.0 -4.8 5.5

#------------------------------------------------------------------------------
# Setup path to data and storage
#------------------------------------------------------------------------------
# Path of current .py file (all data and outputs will be saved here)
file_path = os.path.dirname(get_file_path.__file__)
# SM historical data
#sm_hist_url = 'http://gws-access.jasmin.ac.uk/public/odanceo/tamsat_alert/historical/'
sm_hist_url = 'http://gws-access.jasmin.ac.uk/public/tamsat/tamsat_alert/historical/'
# SM forecast data
#sm_fcast_url = 'http://gws-access.jasmin.ac.uk/public/odanceo/tamsat_alert/forecasts/'
sm_fcast_url = 'http://gws-access.jasmin.ac.uk/public/tamsat/tamsat_alert/forecasts/'
# RFE data
rfe_url = 'http://gws-access.jasmin.ac.uk/public/tamsat/tamsat_alert_forcing_data/subset/Africa/0.25/yearly_files/'

#------------------------------------------------------------------------------
# Setup dates
#------------------------------------------------------------------------------
fcast_date = dtmod.datetime.strptime(str(sys.argv[1]), '%Y%m%d')
poi_start = dtmod.datetime.strptime(str(sys.argv[2]), '%Y%m%d')
poi_end = dtmod.datetime.strptime(str(sys.argv[3]), '%Y%m%d')

met_forc_start_date = poi_start
met_forc_end_date = poi_end

# Define climatological period
clim_start_year = 2005
clim_end_year = 2019

#------------------------------------------------------------------------------
# Setup location (point / bounding box)
#------------------------------------------------------------------------------
if sys.argv[7] == "point":
    lon_point = float(sys.argv[8])
    lat_point = float(sys.argv[9])
    lon_min = "NA"
    lon_max = "NA"
    lat_min = "NA"
    lat_max = "NA"
elif sys.argv[7] == "region":
    lon_point = "NA"
    lat_point = "NA"
    lon_min = float(sys.argv[8])
    lon_max = float(sys.argv[9])
    lat_min = float(sys.argv[10])
    lat_max = float(sys.argv[11])
else:
    raise RuntimeError("argument muct be either 'point' or 'region'")

#------------------------------------------------------------------------------
# Define forecast weights [A,N,B]
#------------------------------------------------------------------------------
weights = np.array([float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])])

if np.sum(weights) != 1:
    raise RuntimeError("tercile weights must add up to 1.0")

#------------------------------------------------------------------------------
# Run forecasts and generate outputs
#------------------------------------------------------------------------------
def forecast_wrapper():
    beta_hist, beta_fcast, forecast_date = import_forecast_data(poi_start, poi_end, fcast_date, sm_hist_url, sm_fcast_url, file_path, clim_start_year, clim_end_year)
    beta_poi, beta_full = splice_forecast(poi_start, poi_end, forecast_date, beta_fcast, beta_hist)
    precip_hist_xr = import_hist_precip(clim_start_year, clim_end_year, rfe_url, file_path)
    beta_hist_xr = import_hist_sm(clim_start_year, clim_end_year, sm_hist_url, file_path)
    beta_fcast_poi_roi, beta_fcast_full_roi, precip_hist_roi, beta_hist_roi = extract_to_roi(beta_poi, beta_full, precip_hist_xr, beta_hist_xr, lon_min, lon_max, lat_min, lat_max, lon_point, lat_point, weights)
    weights_array_masked = weight_forecast(precip_hist_roi, met_forc_start_date, met_forc_end_date, poi_start, poi_end, clim_start_year, clim_end_year, weights)
    beta_hist_full_roi, beta_hist_poi_roi_mean = calc_sm_climatology(beta_hist_roi, clim_start_year, clim_end_year, forecast_date, poi_start, poi_end)
    ens_mean_wrsi_xr, ens_sd_wrsi_xr, clim_mean_wrsi_xr, clim_sd_wrsi_xr, ensemble_forecast = summary_stats(beta_hist_poi_roi_mean, weights_array_masked, beta_fcast_poi_roi, beta_fcast_full_roi)
    forecast_stamp, poi_stamp, poi_str, loc_stamp = date_stamps(forecast_date, poi_start, poi_end, lon_min, lon_max, lat_min, lat_max, lon_point, lat_point)
    output_forecasts(ens_mean_wrsi_xr, ens_sd_wrsi_xr, clim_mean_wrsi_xr, clim_sd_wrsi_xr, ensemble_forecast, beta_hist_full_roi, poi_stamp, forecast_stamp, clim_start_year, clim_end_year, poi_str, forecast_date, loc_stamp)

#------------------------------------------------------------------------------
# Import soil moisture data (historical and forecast)
#------------------------------------------------------------------------------
def import_forecast_data(poi_start, poi_end, forecast_date, sm_hist_url, sm_fcast_url, file_path, clim_start_year, clim_end_year):
    
    # Throw error if poi_end more than 150 days from forecast_date
    if (poi_end - forecast_date).days > 150:
        raise RuntimeError("poi_end beyond the forecast window: cannot forecast beyond 150 days")
    # Throw an error if forecast_date beyond poi_end
    if forecast_date > poi_end:
        raise RuntimeError("forecast date is beyond the end of the poi")
    
    # Identify which forecast file is needed
    fcast_files = get_all_files(sm_fcast_url)
    str_match = re.compile('alert_(.+?)_ens.daily.nc')
    fcast_files = list(filter(str_match.match, fcast_files))
    fcast_dates = []
    for i in np.arange(0, len(fcast_files)):
        fdate_str = re.search('alert_(.+?)_ens.daily.nc', fcast_files[i]).group(1)
        fdate = dtmod.datetime.strptime(fdate_str, "%Y%m%d")
        fcast_dates.append(fdate)
    fcast_file = min(fcast_dates, key=lambda x: abs(x - forecast_date))
    fcast_stamp = fcast_file.strftime("%Y%m%d")
    
    # Reset forecast date depending on latest available forecast data
    forecast_date = dtmod.datetime.strptime(fcast_stamp, "%Y%m%d")
    
    # Import relevant forecast file
    url = sm_fcast_url + 'alert_' + fcast_stamp + '_ens.daily.nc'
    fname = file_path + '/forecasts/alert_' + fcast_stamp + '_ens.daily.nc'
    if os.path.isfile(fname) == True:
        if Path(fname).stat().st_size < 3652942409:
            os.remove(fname)
            print("Retrieving forecast files... This can take several minutes")
            urllib.request.urlretrieve(url, fname)
    if os.path.isfile(fname) == False:
        print("Retrieving forecast files... This can take several minutes")
        urllib.request.urlretrieve(url, fname)
    fcast_df = xr.open_dataset(fname)
    fcast_df = fcast_df.assign_coords({"ens_year": np.arange(clim_start_year, clim_end_year+1, 1)})
        
    # Identify which historical files (if any) are needed (deals with year boundary)
    if forecast_date <= poi_start: # if the forecast date is before the start of the poi
        hist_df_yr = [poi_start.year] # no historical data needed, but use holder to make following code work
    elif poi_start.year == poi_end.year: # if the poi is contained within one year
        hist_df_yr = [poi_start.year] # no need to worry about the year boundary
    elif poi_start.year == forecast_date.year: # if the poi start and forecast date are in the same year
        hist_df_yr = [poi_start.year] # no need to worry about the year boundary (even if poi end is in next year - dealt with below)
    else: # if poi start and forecast date are in different years
        hist_df_yr = [poi_start.year, forecast_date.year] # you need historical data for both years
    
    # Import relevant historical data (if required)
    hist_df = []
    for i in np.arange(0, len(hist_df_yr)):
        url = sm_hist_url + 'sm_data.daily.' + str(hist_df_yr[i]) + '.nc'
        fname = file_path + '/historical/sm_data.daily.' + str(hist_df_yr[i]) + '.nc'
        if os.path.isfile(fname) == True:
            os.remove(fname)
        print("Updating historic files...")
        urllib.request.urlretrieve(url, fname)
        hist_df.append(xr.open_dataset(fname))
    # Concatenate historical data if the historic period spans the year boundary
    if len(hist_df) > 1:
        hist_df = xr.concat([hist_df[0], hist_df[1]], "time")
    
    # Extract soil moisture variable of interest
    beta_hist = hist_df[0]["beta_c4grass"]
    beta_fcast = fcast_df["beta_c4grass"]
    # Remove weird values in data ###NEED TO ASK EWAN ABOUT THIS!
    beta_hist = beta_hist.where(beta_hist != 0.0, np.nan)
    beta_fcast = beta_fcast.where(beta_fcast != 0.0, np.nan)
        
    # Sort out hist datetime issue
    time00 = pd.to_datetime(beta_hist["time"].values) - dtmod.timedelta(hours = 12)
    beta_hist = beta_hist.assign_coords(time = time00)
    
    return beta_hist, beta_fcast, forecast_date

# Identify closest forecast file to forecast date specified
def get_all_files(sm_fcast_url):
    soup = BeautifulSoup(requests.get(sm_fcast_url).text, features="lxml")
    fcast_files = []
    for a in soup.findAll(href = re.compile("\.nc$")):
        fcast_files.append(a['href'])
    return fcast_files

#------------------------------------------------------------------------------
# Splice data together for poi
#------------------------------------------------------------------------------
def splice_forecast(poi_start, poi_end, forecast_date, beta_fcast, beta_hist):
    
    if forecast_date <= poi_start:
        beta_full = beta_fcast.loc[dict(time = slice(forecast_date, poi_end))] # includes run up to season if forecast before season - for plotting later
        beta_full = beta_full.transpose("longitude","latitude","time","ens_year")
        beta_poi = beta_fcast.loc[dict(time = slice(poi_start, poi_end))]
        beta_poi = beta_poi.transpose("longitude","latitude","time","ens_year")
    else:
        hist_splice = beta_hist.loc[dict(time = slice(poi_start, forecast_date - dtmod.timedelta(days = 1)))]
        #hist_splice = hist_splice.groupby("time").mean(skipna = True)
        fcast_splice = beta_fcast.loc[dict(time = slice(forecast_date, poi_end))]
        beta_poi = xr.merge(xr.broadcast(hist_splice, fcast_splice))["beta_c4grass"] ###CHECK ["beta_c4grass"] is needed
        beta_poi = beta_poi.transpose("longitude","latitude","time","ens_year")
        beta_full = beta_poi # for plotting later
        
    return beta_poi, beta_full

#------------------------------------------------------------------------------
# Import historical precip data (for weighting)
#------------------------------------------------------------------------------
def import_hist_precip(clim_start_year, clim_end_year, rfe_url, file_path):
    
    # Define climatological period
    clim_years = np.arange(clim_start_year, clim_end_year + 1, 1)
    # Choose rainfall dataset: 'tamsat' or 'chirps'
    rain_dataset = 'tamsat'
    # Create empty array to store precip file names
    precip_filenames = []
    # List precip files (one per year in climatological period)
    for yr in clim_years:
        if rain_dataset == 'tamsat':
            url = rfe_url + str(yr) + '/prate_tamsat_' + str(yr) + '_sub.nc'
            fname = file_path + '/rfe/prate_tamsat_' + str(yr) + '_sub.nc'
            if os.path.isfile(fname) == False:
                print('Retrieving precip files... This can take some time...' + str(yr))
                urllib.request.urlretrieve(url, fname)
            precip_filenames.append(fname)
        if rain_dataset == 'chirps':
            raise RuntimeError("Not working with CHIRPS yet: please use TAMSAT")
            precip_filenames.append(str('F:/Ewan_SM_data_260121/rfe/chirps-v2.sub.')+str(yr)+str(".nc"))
    # Load all precip data as single xarray
    precip_hist_xr = xr.open_mfdataset(precip_filenames)
    
    # If using TAMSAT rainfall data, create 'precip' variable to streamline future code
    if rain_dataset == 'tamsat':
        precip_hist_xr['precip'] = precip_hist_xr['rfe']
        precip_hist_xr = precip_hist_xr.drop(['rfe'])
        
    return precip_hist_xr

#------------------------------------------------------------------------------
# Import historical soil moisture data (for climatology)
#------------------------------------------------------------------------------
def import_hist_sm(clim_start_year, clim_end_year, sm_hist_url, file_path):
    
    # Define climatological period
    clim_years = np.arange(clim_start_year, clim_end_year + 1, 1)
    # Create empty array to store sm file names
    beta_filenames = []
    # List precip files (one per year in climatological period)
    for yr in clim_years:
        url = sm_hist_url + 'sm_data.daily.' + str(yr) + '.nc'
        fname = file_path + '/historical/sm_data.daily.' + str(yr) + '.nc'
        if os.path.isfile(fname) == False:
            print('Retrieving historical soil moisture files... This can take some time...' + str(yr))
            urllib.request.urlretrieve(url, fname)
        beta_filenames.append(fname)
    # Load all precip data as single xarray
    beta_hist_xr = xr.open_mfdataset(beta_filenames)
    beta_hist_xr = beta_hist_xr['beta_c4grass']
    # Remove weird values in data
    beta_hist_xr = beta_hist_xr.where(beta_hist_xr != 0.0, np.nan)
    
    return beta_hist_xr

#------------------------------------------------------------------------------
# Extract to lat-lon point / region
#------------------------------------------------------------------------------
def extract_to_roi(beta_poi, beta_full, precip_hist_xr, beta_hist_xr, lon_min, lon_max, lat_min, lat_max, lon_point, lat_point, weights):
    
    lons = beta_poi["longitude"]
    lats = beta_poi["latitude"]
    
    if lon_point == "NA": # if considering a region
        lon_min_ind = find_lon_lat_ind(lons, lon_min)
        lon_max_ind = find_lon_lat_ind(lons, lon_max)
        lat_min_ind = find_lon_lat_ind(lats, lat_min)
        lat_max_ind = find_lon_lat_ind(lats, lat_max)
    else: # if considering a point
        lon_min_ind = find_lon_lat_ind(lons, lon_point)
        lon_max_ind = find_lon_lat_ind(lons, lon_point) # duplicated because we need the min and max for the splice
        lat_min_ind = find_lon_lat_ind(lats, lat_point)
        lat_max_ind = find_lon_lat_ind(lats, lat_point)
    
    # Crop data to region / point of interest
    beta_fcast_poi_roi = beta_poi.sel(longitude = slice(lons[lon_min_ind],lons[lon_max_ind]), latitude = slice(lats[lat_min_ind],lats[lat_max_ind]))
    beta_fcast_full_roi = beta_full.sel(longitude = slice(lons[lon_min_ind],lons[lon_max_ind]), latitude = slice(lats[lat_min_ind],lats[lat_max_ind]))
    
    precip_hist_roi = precip_hist_xr.sel(longitude = slice(lons[lon_min_ind],lons[lon_max_ind]), latitude = slice(lats[lat_min_ind],lats[lat_max_ind]))
    beta_hist_roi = beta_hist_xr.sel(longitude = slice(lons[lon_min_ind],lons[lon_max_ind]), latitude = slice(lats[lat_min_ind],lats[lat_max_ind]))
    
    # If working with spatially varying weighting ###NOT FUNCTIONAL YET!
    if len(weights.shape) > 1:
        weights_roi = weights.sel(longitude = slice(lons[lon_min_ind],lons[lon_max_ind]), latitude = slice(lats[lat_min_ind],lats[lat_max_ind]))

    return beta_fcast_poi_roi, beta_fcast_full_roi, precip_hist_roi, beta_hist_roi

# Function: find indices relating to region / point of interest
def find_lon_lat_ind(array, point):
    # Loop through lon and lat values to identify correct grid cell
    for i in np.arange(0,len(array),1):
        if array[i] > point:
            ind = int(np.where(array==array[i-1])[0])
            break
    return ind
#------------------------------------------------------------------------------
# Weighting
#------------------------------------------------------------------------------
def weight_forecast(precip_hist_roi, met_forc_start_date, met_forc_end_date, poi_start, poi_end, clim_start_year, clim_end_year, weights):
    
    # Define climatological period
    clim_years = np.arange(clim_start_year, clim_end_year + 1, 1)
    
    # Reshape precip data so that we can splice out the poi (and deal with year boundary)
    precip_hist_roi_reshape = reshape_hist_data(precip_hist_roi['precip'].transpose('longitude','latitude','time').values, clim_start_year)
    precip_hist_roi_reshape = make_two_year_array(precip_hist_roi_reshape)[:,:,:,0:len(clim_years)]
    
    # Remove any dodgey precip values < 0
    precip_hist_roi_reshape[precip_hist_roi_reshape < 0] = np.nan
    
    # Assign coordinates for xarray
    longitude = precip_hist_roi['longitude'].values
    latitude = precip_hist_roi['latitude'].values
    times = np.arange(0,730,1) # just days for now, convert to dates in a min
    
    # Convert back to xarray
    precip_hist_roi_reshape_xr = xr.DataArray(precip_hist_roi_reshape,
                                              coords = [longitude, latitude, times, clim_years],
                                              dims = ['longitude','latitude','time','ens_year'])
    
    # Fill dates and splice to period to which weighting applies
    precip_hist_roi_reshape_xr = precip_hist_roi_reshape_xr.assign_coords({"time": pd.date_range(start = dtmod.datetime(poi_start.year,1,1), end = dtmod.datetime(poi_start.year+1,12,31))}) # dates only for current year
    precip_poi_roi = precip_hist_roi_reshape_xr.sel(time = slice(met_forc_start_date, met_forc_end_date))
       
    # Calculate annual mean precip for poi 
    precip_poi_roi_mean = precip_poi_roi.mean(axis = 2, skipna = True)
    
    # If weights vary spatially, interpolate precip to same grid as forecast ###THIS WIlL NEED TESTING
    if len(weights.shape) > 1:
        print("interpolating precip to weighting array")
        lonsout = weights['longitude'].values
        latsout = weights['latitude'].values
        precip_poi_roi_mean = precip_poi_roi_mean.interp(longitude = lonsout, latitude = latsout)
    
    # If using uniform weighting
    if len(weights.shape) == 1: 
        t1 = weights[0]
        t2 = weights[1]
        t3 = weights[2]
    
    # Calculate climatological mean and standard deviation for each lon-lat grid cell
    precip_clim_mean = np.nanmean(precip_poi_roi_mean, axis = 2)
    precip_clim_sd = np.nanstd(precip_poi_roi_mean, axis = 2)
    
    # Calculate tercile boundaries for each lon-lat grid cell
    t1_thres = scipy.stats.norm(precip_clim_mean, precip_clim_sd).ppf(0.33)
    t2_thres = scipy.stats.norm(precip_clim_mean, precip_clim_sd).ppf(0.67)
    
    if lon_max != "NA":
        t1_thres = np.repeat(np.expand_dims(t1_thres, 2), repeats = precip_poi_roi_mean.shape[2], axis = 2)
        t2_thres = np.repeat(np.expand_dims(t2_thres, 2), repeats = precip_poi_roi_mean.shape[2], axis = 2)
    
    # Fill in weights for each year and lon-lat grid cell
    precip_weights = precip_poi_roi_mean.where(precip_poi_roi_mean > t1_thres, t3*1e06)
    precip_weights = precip_weights.where(precip_weights > t2_thres, t2*1e06)
    precip_weights = precip_weights.where(precip_weights > 1000, t1*1e06)

    ###HOW TO APPLY SPATIaLLY VARYING WIEGHTING?
    
    # Convert to DataArray
    weights_array = xr.DataArray(precip_weights/1e06,
                                 coords = [longitude,latitude,clim_years],
                                 dims = ['longitude','latitude','ens_year']).to_dataset(name = 'weights')
    
    # Mask nan values
    weights_array_masked = np.ma.MaskedArray(weights_array['weights'].values, mask = np.isnan(weights_array['weights'].values))
    
    return weights_array_masked

#------------------------------------------------------------------------------
# SM climatology
#------------------------------------------------------------------------------
def calc_sm_climatology(beta_hist_roi, clim_start_year, clim_end_year, forecast_date, poi_start, poi_end, ):
    
    # Define climatological period
    clim_years = np.arange(clim_start_year, clim_end_year + 1, 1)
    # Reshape sm hist data so that we can splice out the poi (and deal with year boundary)
    beta_hist_roi_reshape = reshape_hist_data(beta_hist_roi.transpose('longitude','latitude','time').values, clim_start_year)
    beta_hist_roi_reshape = make_two_year_array(beta_hist_roi_reshape)[:,:,:,0:len(clim_years)]
    
    # Assign coordinates for xarray
    longitude = beta_hist_roi['longitude'].values
    latitude = beta_hist_roi['latitude'].values
    times = np.arange(0,730,1) # just days for now, convert to dates in a min
    
    # Convert back to xarray
    beta_hist_roi_reshape_xr = xr.DataArray(beta_hist_roi_reshape,
                                            coords = [longitude, latitude, times, clim_years],
                                            dims = ['longitude','latitude','time','ens_year'])
    # Fill dates and splice to period to which weighting applies
    beta_hist_roi_reshape_xr = beta_hist_roi_reshape_xr.assign_coords({"time":pd.date_range(start = dtmod.datetime(poi_start.year,1,1), end = dtmod.datetime(poi_start.year+1,12,31))})
    beta_hist_poi_roi = beta_hist_roi_reshape_xr.sel(time = slice(poi_start, poi_end))
    
    # Splice to full period - if forecast date before poi start
    if forecast_date <= poi_start:
        beta_hist_full_roi = beta_hist_roi_reshape_xr.sel(time = slice(forecast_date, poi_end))
    else:
        beta_hist_full_roi = beta_hist_poi_roi
        
    # Calculate annual mean precip for poi 
    beta_hist_poi_roi_mean = beta_hist_poi_roi.mean(axis = 2, skipna = True)
    
    return beta_hist_full_roi, beta_hist_poi_roi_mean

#------------------------------------------------------------------------------
# Summary statistics
#------------------------------------------------------------------------------
def summary_stats(beta_hist_poi_roi_mean, weights_array_masked, beta_fcast_poi_roi, beta_fcast_full_roi):
    
    longitude = beta_hist_poi_roi_mean['longitude']
    latitude = beta_hist_poi_roi_mean['latitude']

    # Calculate climatological mean and standard deviation from historical data 
    clim_mean_wrsi = np.average(beta_hist_poi_roi_mean, axis = 2) 
    av = np.repeat(np.expand_dims(clim_mean_wrsi, axis = 2), repeats = beta_hist_poi_roi_mean.shape[2], axis = 2)
    clim_sd_wrsi = np.sqrt(np.average(np.abs(beta_hist_poi_roi_mean - av), axis = 2)) 
    
    # Calculate forecast mean beta
    beta_fcast_poi_roi_mean = np.nanmean(beta_fcast_poi_roi, axis = 2)
    
    # Calculate ensemble mean and standard deviation
    ens_mean_wrsi = np.average(beta_fcast_poi_roi_mean, weights = weights_array_masked, axis = -1) 
    av = np.repeat(np.expand_dims(ens_mean_wrsi, axis = -1), repeats = beta_fcast_poi_roi_mean.shape[-1], axis = -1)
    ens_sd_wrsi = np.sqrt(np.average(np.abs(beta_fcast_poi_roi_mean - av), weights = weights_array_masked, axis = -1))
    
    # Ensemble forecast - already an xarray
    ensemble_forecast = beta_fcast_full_roi # includes run up to season if forecast before season - for plotting
    
    # Convert to xarrays
    ens_mean_wrsi_xr = xr.DataArray(ens_mean_wrsi, coords = [longitude,latitude], dims = ['longitude','latitude'])
    ens_sd_wrsi_xr = xr.DataArray(ens_sd_wrsi, coords = [longitude,latitude], dims = ['longitude','latitude'])
    clim_mean_wrsi_xr = xr.DataArray(clim_mean_wrsi, coords = [longitude,latitude], dims = ['longitude','latitude'])
    clim_sd_wrsi_xr = xr.DataArray(clim_sd_wrsi, coords = [longitude,latitude], dims = ['longitude','latitude'])
    
    return ens_mean_wrsi_xr, ens_sd_wrsi_xr, clim_mean_wrsi_xr, clim_sd_wrsi_xr, ensemble_forecast

#------------------------------------------------------------------------------
# Date stamps for output files
#------------------------------------------------------------------------------
def date_stamps(forecast_date, poi_start, poi_end, lon_min, lon_max, lat_min, lat_max, lon_point, lat_point):

    # Forecast date stamps
    forecast_stamp = forecast_date.strftime("%Y%m%d")
    
    # POI stamps
    start_month = poi_start.month
    end_month = poi_end.month
    poi_months = np.arange(start_month, end_month + 1, 1)
    poi_year = poi_start.year
    
    poi_str = ""
    for mo in np.arange(0,len(poi_months)):
        tmp_date = dtmod.datetime(2020,poi_months[mo],1).strftime("%b")[0]
        poi_str += tmp_date
    
    poi_stamp = poi_str + str(poi_year)
    
    if lon_point != "NA":
        loc_stamp = str(lon_point) + "_" + str(lat_point)
    else:
        loc_stamp = str(lon_min) + "_" + str(lon_max) + "_" + str(lat_min) + "_" + str(lat_max)
    
    return forecast_stamp, poi_stamp, poi_str, loc_stamp

#------------------------------------------------------------------------------
# Outputs
#------------------------------------------------------------------------------
def output_forecasts(ens_mean_wrsi_xr, ens_sd_wrsi_xr, clim_mean_wrsi_xr, clim_sd_wrsi_xr, ensemble_forecast, beta_hist_full_roi, poi_stamp, forecast_stamp, clim_start_year, clim_end_year, poi_str, forecast_date, loc_stamp):
    
    # Save output files
    ens_mean_wrsi_xr.to_netcdf(file_path+"/outputs/ens_mean_wrsi_"+poi_stamp+"_"+forecast_stamp+"_"+loc_stamp+".nc")
    ens_sd_wrsi_xr.to_netcdf(file_path+"/outputs/ens_sd_wrsi_"+poi_stamp+"_"+forecast_stamp+"_"+loc_stamp+".nc")
    clim_mean_wrsi_xr.to_netcdf(file_path+"/outputs/clim_mean_wrsi_"+poi_stamp+"_"+forecast_stamp+"_"+loc_stamp+".nc")
    clim_sd_wrsi_xr.to_netcdf(file_path+"/outputs/clim_sd_wrsi_"+poi_stamp+"_"+forecast_stamp+"_"+loc_stamp+".nc")
    ensemble_forecast.to_netcdf(file_path+"/outputs/ensemble_forecast_wrsi_"+poi_stamp+"_"+forecast_stamp+"_"+loc_stamp+".nc")
    
    terciles_text(clim_mean_wrsi_xr, clim_sd_wrsi_xr, ens_mean_wrsi_xr, ens_sd_wrsi_xr, poi_stamp, forecast_stamp, loc_stamp)
    prob_dist_plot(clim_mean_wrsi_xr, clim_sd_wrsi_xr, ens_mean_wrsi_xr, ens_sd_wrsi_xr, poi_stamp, forecast_stamp, loc_stamp)
    ensemble_timeseries_plot(ensemble_forecast, forecast_date, poi_start, poi_end, beta_hist_full_roi, poi_stamp, forecast_stamp, loc_stamp)    
    
    if sys.argv[7] == "region":
        anom_map_plot(clim_mean_wrsi_xr, ens_mean_wrsi_xr, poi_stamp, forecast_stamp, clim_start_year, clim_end_year, poi_str, loc_stamp)
        prob_map_plot(clim_mean_wrsi_xr, clim_sd_wrsi_xr, ens_mean_wrsi_xr, ens_sd_wrsi_xr, poi_stamp, forecast_stamp, loc_stamp)


# Calculate tercile probabilities
def terciles_text(clim_mean_wrsi_xr, clim_sd_wrsi_xr, ens_mean_wrsi_xr, ens_sd_wrsi_xr, poi_stamp, forecast_stamp, loc_stamp):
    # Calculate probability of lower tercile soil moisture
    a = scipy.stats.norm(clim_mean_wrsi_xr, clim_sd_wrsi_xr).ppf(0.33)
    b_lower = scipy.stats.norm(ens_mean_wrsi_xr, ens_sd_wrsi_xr).cdf(a)
    # Calculate probability of mid tercile soil moisture
    a = scipy.stats.norm(clim_mean_wrsi_xr, clim_sd_wrsi_xr).ppf(0.67)
    b_upper = scipy.stats.norm(ens_mean_wrsi_xr, ens_sd_wrsi_xr).cdf(a)
    # Calculate all terciles
    lower_terc = b_lower
    middle_terc = b_upper - b_lower
    upper_terc = 1 - b_upper
    # Calculate mean of all terciles
    lower_terc = np.nanmean(lower_terc)
    middle_terc = np.nanmean(middle_terc)
    upper_terc = np.nanmean(upper_terc)
    # Print and save tercile probabilities
    text_out = open(file_path+"/outputs/terciles_"+poi_stamp+"_"+forecast_stamp+"_"+loc_stamp+".txt","w")
    text_out.write("Probability of seasonal mean soil moisture (beta) falling into the following terciles:\nLower: "+str(round(lower_terc, 2))+", Middle: "+str(round(middle_terc, 2))+", Upper: "+str(round(upper_terc, 2)))
    text_out.close()
    print("Lower: "+str(round(lower_terc, 2))+", Middle: "+str(round(middle_terc, 2))+", Upper: "+str(round(upper_terc, 2)))

# Plot probability distributions
def prob_dist_plot(clim_mean_wrsi_xr, clim_sd_wrsi_xr, ens_mean_wrsi_xr, ens_sd_wrsi_xr, poi_stamp, forecast_stamp, loc_stamp):
    # Define tercile boundaries
    lower_thresh = 0.33
    upper_thresh = 0.67
    # Create probability distribution of climatology
    clim_dist = np.random.normal(loc = np.nanmean(clim_mean_wrsi_xr), scale = np.nanmean(clim_sd_wrsi_xr), size = 10000)
    # Create probability distribution of ensemble forecast
    ens_dist = np.random.normal(loc = np.nanmean(ens_mean_wrsi_xr), scale = np.nanmean(ens_sd_wrsi_xr), size = 10000)
    # Calculate tercile thresholds
    low_a = np.nanmean(scipy.stats.norm(np.nanmean(clim_mean_wrsi_xr), np.nanmean(clim_sd_wrsi_xr)).ppf(lower_thresh))
    up_a = np.nanmean(scipy.stats.norm(np.nanmean(clim_mean_wrsi_xr), np.nanmean(clim_sd_wrsi_xr)).ppf(upper_thresh))
    # Convert to pandas series - for plotting purposes
    clim_dist_pd = pd.Series(clim_dist)
    ens_dist_pd = pd.Series(ens_dist)
    # Get plotting parameters - x axis limits
    clim_dist_pd.plot.hist()
    ens_dist_pd.plot.hist()
    ax = plt.gca()
    xlims = ax.get_xlim()
    plt.close()
    # Build plot
    plt.figure(figsize = (6,4))
    clim_dist_pd.plot.density(color = "black", linewidth = 2, xlim = (xlims), label = "Climatology")
    ens_dist_pd.plot.density(color = "red", linewidth = 2, label = "Forecast")
    plt.xlabel("Seasonal mean SM (beta)", fontweight = "bold")
    plt.ylabel("Probability", fontweight = "bold")
    plt.axvline(low_a, color = "grey", linestyle = "--", label = "Tercile boundaries")
    plt.axvline(up_a, color = "grey", linestyle = "--")
    plt.legend(loc = 2)
    # Save plot
    plt.savefig(file_path+"/outputs/probdist_"+poi_stamp+"_"+forecast_stamp+"_"+loc_stamp+".png")
    plt.close()
    
# Plot ensemble forecast compared to climatology
def ensemble_timeseries_plot(ensemble_forecast, forecast_date, poi_start, poi_end, beta_hist_full_roi, poi_stamp, forecast_stamp, loc_stamp):
    # Create data frame of dates
    date_labs = pd.to_datetime(ensemble_forecast['time'].values)
    # Setup plot
    plt.figure(figsize = (7,4))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval = 14))
    # Depending on positioning of forecast date relative to poi
    if forecast_date <= poi_start: # If all forecast
        plt.plot(date_labs, np.nanmean(ensemble_forecast, axis = (0,1)), color = "grey", label = "Ensemble forecast")   
        plt.plot(forecast_date, np.nanmean(ensemble_forecast.sel(time = forecast_date)), 
                 marker = "o", color = "red", markersize = 8, label = "Forecast date")
    else: # If some observed and some forecast
        obs = ensemble_forecast.sel(time = slice(poi_start, forecast_date - dtmod.timedelta(days = 1)))
        fcast = ensemble_forecast.sel(time = slice(forecast_date - dtmod.timedelta(days = 1), poi_end))
        date_obs = pd.to_datetime(obs["time"].values)
        date_fcast = pd.to_datetime(fcast["time"].values)
        plt.plot(date_obs, np.nanmean(obs, axis = (0,1)), color = "black", label = "Observed")
        plt.plot(date_fcast, np.nanmean(fcast, axis = (0,1)), color = "grey", label = "Ensemble forecast")
        plt.plot(forecast_date - dtmod.timedelta(days = 1), np.nanmean(ensemble_forecast.sel(time = forecast_date)),
                 marker = "o", color = "red", markersize = 8, label = "Forecast date")    
    # Continue with plotting visuals
    plt.fill_between(date_labs, np.nanpercentile(np.nanmean(beta_hist_full_roi, axis =(0,1)), 5, axis = (1)), np.nanpercentile(np.nanmean(beta_hist_full_roi, axis = (0,1)), 95, axis = (1)), 
                     alpha = 0.35, color = "grey", label = "5th-95th percentile")
    plt.axvline(poi_start, color = "black", linestyle = "--", label = "POI boundaries")
    plt.axvline(poi_end, color = "black", linestyle = "--")
    plt.ylabel("Soil moisture (beta)", fontweight = "bold")
    plt.gcf().autofmt_xdate()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc = 2)
    plt.savefig(file_path+"/outputs/timeseries_"+poi_stamp+"_"+forecast_stamp+"_"+loc_stamp+".png")
    plt.close()

# Plot forecast anomaly map
def anom_map_plot(clim_mean_wrsi_xr, ens_mean_wrsi_xr, poi_stamp, forecast_stamp, clim_start_year, clim_end_year, poi_str, loc_stamp):
    # Extract lons and lats for plotting axies
    lons = clim_mean_wrsi_xr['longitude'].values
    lats = clim_mean_wrsi_xr['latitude'].values
    # Calculate max values to standardised colorbars on both plots
    vmax = np.nanmax([clim_mean_wrsi_xr, ens_mean_wrsi_xr])
    # Calculate percent anomaly    
    percent_anom = (ens_mean_wrsi_xr / clim_mean_wrsi_xr) * 100
    # Save to netCDF - perc_anom
    percent_anom_xr = xr.DataArray(percent_anom, coords = [lons,lats], dims = ['longitude','latitude'])
    percent_anom_xr.to_netcdf(file_path+"/outputs/percent_anom_"+poi_stamp+"_"+forecast_stamp+"_"+loc_stamp+".nc")
    # Colormap setup - make 'bad' values grey
    BrBG_cust = matplotlib.cm.get_cmap("BrBG")
    BrBG_cust.set_bad(color = "silver")
    RdBu_cust = matplotlib.cm.get_cmap("RdBu")
    RdBu_cust.set_bad(color = "silver")
    # Build plot
    fig = plt.figure(figsize = (32,10))
    # Plot climatology
    clim_plt = fig.add_subplot(131, projection = ccrs.PlateCarree())
    clim_plt.set_extent([np.min(lons), np.max(lons), np.min(lats), np.max(lats)])
    clim_plt.pcolormesh(lons, lats, clim_mean_wrsi_xr.T, vmin = 0, vmax = vmax, cmap = BrBG_cust)
    clim_gl = clim_plt.gridlines(crs = ccrs.PlateCarree(), draw_labels = True, linewidth = 0.7, color = "black", alpha = 1, linestyle = "--")
    clim_gl.xlabels_top = False
    clim_gl.ylabels_right = False
    clim_gl.xlabel_style = {'size': 18}
    clim_gl.ylabel_style = {'size': 18}
    clim_gl.xformatter = LONGITUDE_FORMATTER
    clim_gl.yformatter = LATITUDE_FORMATTER
    clim_plt.set_title('SM (beta) climatology\n' + poi_str + ' ' + str(clim_start_year) + '-' + str(clim_end_year), fontsize = 20)
    clim_cb = plt.pcolormesh(lons, lats, clim_mean_wrsi_xr.T, vmin = 0, vmax = vmax, cmap = BrBG_cust)
    clim_cb = plt.colorbar(clim_cb)
    clim_cb.ax.tick_params(labelsize=18)
    clim_plt.set_aspect("auto", adjustable = None)
    clim_plt.add_feature(cfeature.OCEAN, facecolor = "white", zorder = 1)
    clim_plt.add_feature(cfeature.COASTLINE, linewidth = 2)
    clim_plt.add_feature(cfeature.BORDERS, linewidth = 2)
    # Plot forecast
    ens_plt = fig.add_subplot(132, projection = ccrs.PlateCarree())
    ens_plt.set_extent([np.min(lons), np.max(lons), np.min(lats), np.max(lats)])
    ens_plt.pcolormesh(lons, lats, ens_mean_wrsi_xr.T, vmin = 0, vmax = vmax, cmap = BrBG_cust)
    ens_gl = ens_plt.gridlines(crs = ccrs.PlateCarree(), draw_labels = True, linewidth = 0.7, color = "black", alpha = 1, linestyle = "--")
    ens_gl.xlabels_top = False
    ens_gl.ylabels_right = False
    ens_gl.xlabel_style = {'size': 18}
    ens_gl.ylabel_style = {'size': 18}
    ens_gl.xformatter = LONGITUDE_FORMATTER
    ens_gl.yformatter = LATITUDE_FORMATTER
    ens_plt.set_title('SM (beta) forecast for ' + poi_stamp + "\nIssued "+ forecast_stamp, fontsize = 20)
    ens_cb = plt.pcolormesh(lons, lats, ens_mean_wrsi_xr.T, vmin = 0, vmax = vmax, cmap = BrBG_cust)
    ens_cb = plt.colorbar(ens_cb)
    ens_cb.ax.tick_params(labelsize=18)
    ens_plt.set_aspect("auto", adjustable = None)
    ens_plt.add_feature(cfeature.OCEAN, facecolor = "white", zorder = 1)
    ens_plt.add_feature(cfeature.COASTLINE, linewidth = 2)
    ens_plt.add_feature(cfeature.BORDERS, linewidth = 2)
    # Plot anomaly
    anom_plt = fig.add_subplot(133, projection = ccrs.PlateCarree())
    anom_plt.set_extent([np.min(lons), np.max(lons), np.min(lats), np.max(lats)])
    anom_plt.pcolormesh(lons, lats, percent_anom.T, vmin = 50, vmax = 150, cmap = RdBu_cust)
    anom_gl = anom_plt.gridlines(crs = ccrs.PlateCarree(), draw_labels = True, linewidth = 0.7, color = "black", alpha = 1, linestyle = "--")
    anom_gl.xlabels_top = False
    anom_gl.ylabels_right = False
    anom_gl.xlabel_style = {'size': 18}
    anom_gl.ylabel_style = {'size': 18}
    anom_gl.xformatter = LONGITUDE_FORMATTER
    anom_gl.yformatter = LATITUDE_FORMATTER
    anom_plt.set_title('SM (beta) % anomaly for ' + poi_stamp + "\nIssued "+ forecast_stamp, fontsize = 20)
    anom_cb = plt.pcolormesh(lons, lats, percent_anom.T, vmin = 50, vmax = 150, cmap = RdBu_cust)
    anom_cb = plt.colorbar(anom_cb)
    anom_cb.ax.tick_params(labelsize=18)
    anom_plt.set_aspect("auto", adjustable = None)
    anom_plt.add_feature(cfeature.OCEAN, facecolor = "white", zorder = 1)
    anom_plt.add_feature(cfeature.COASTLINE, linewidth = 2)
    anom_plt.add_feature(cfeature.BORDERS, linewidth = 2)
    # Save and show
    plt.savefig(file_path+"/outputs/map_plot"+poi_stamp+"_"+forecast_stamp+"_"+loc_stamp+".png")
    plt.close()

# Plot probability of lower tercile map
def prob_map_plot(clim_mean_wrsi_xr, clim_sd_wrsi_xr, ens_mean_wrsi_xr, ens_sd_wrsi_xr, poi_stamp, forecast_stamp, loc_stamp):
    # Extract lons and lats for plotting axies
    lons = clim_mean_wrsi_xr['longitude'].values
    lats = clim_mean_wrsi_xr['latitude'].values
    lower_thresh = 0.33
    # Calculate probability of lower tercile soil moisture
    a = scipy.stats.norm(clim_mean_wrsi_xr, clim_sd_wrsi_xr).ppf(0.33)
    b_lower = scipy.stats.norm(ens_mean_wrsi_xr, ens_sd_wrsi_xr).cdf(a)
    # Save to netCDF - prob_lower_terc
    b_lower_xr = xr.DataArray(b_lower, coords = [lons,lats], dims = ['longitude','latitude'])
    b_lower_xr.to_netcdf(file_path+"/outputs/prob_lower_tercile_"+poi_stamp+"_"+forecast_stamp+"_"+loc_stamp+".nc")
   # Colormap setup - make 'bad' values grey
    c = mcolors.ColorConverter().to_rgb
    rvb = make_colormap([c('green'), c('palegreen'), lower_thresh - 0.05, c('white'), c('white'), lower_thresh + 0.05, c('yellow'), c('brown')])
    rvb_cust = matplotlib.cm.get_cmap(rvb)
    rvb_cust.set_bad(color = "silver")
    # Extract lons and lats for plotting axies
    lons = clim_mean_wrsi_xr['longitude'].values
    lats = clim_mean_wrsi_xr['latitude'].values    
    # Build plot
    fig = plt.figure(figsize = (10,10))
    # Plot climatology
    prob_plt = fig.add_subplot(111, projection = ccrs.PlateCarree())
    prob_plt.set_extent([np.min(lons), np.max(lons), np.min(lats), np.max(lats)])
    prob_plt.pcolormesh(lons, lats, b_lower.T, vmin = 0, vmax = 1, cmap = rvb_cust)
    prob_gl = prob_plt.gridlines(crs = ccrs.PlateCarree(), draw_labels = True, linewidth = 0.7, color = "black", alpha = 1, linestyle = "--")
    prob_gl.xlabels_top = False
    prob_gl.ylabels_right = False
    prob_gl.xlabel_style = {'size': 18}
    prob_gl.ylabel_style = {'size': 18}
    prob_gl.xformatter = LONGITUDE_FORMATTER
    prob_gl.yformatter = LATITUDE_FORMATTER
    prob_plt.set_title('Probability of lower tercile SM\n' + poi_stamp + " Issued "+ forecast_stamp, fontsize = 20)
    prob_cb = plt.pcolormesh(lons, lats, b_lower.T, vmin = 0, vmax = 1, cmap = rvb_cust)
    prob_cb = plt.colorbar(prob_cb)
    prob_cb.ax.tick_params(labelsize=18)
    prob_plt.set_aspect("auto", adjustable = None)
    prob_plt.add_feature(cfeature.OCEAN, facecolor = "white", zorder = 1)
    prob_plt.add_feature(cfeature.COASTLINE, linewidth = 2)
    prob_plt.add_feature(cfeature.BORDERS, linewidth = 2)
    # Save and show
    plt.savefig(file_path+"/outputs/prob_map_plot"+poi_stamp+"_"+forecast_stamp+"_"+loc_stamp+".png")
    plt.close()

#------------------------------------------------------------------------------
# sm_gridded_utils functions
#------------------------------------------------------------------------------
def reshape_hist_data(datain,startyear):
    '''
    This function reorganises a historical daily time series (long,lat,time array) into an array with year per row. 
    It is assumed that the data start on January 1st. Leap days are removed.
    param datain: daily time series
    param startyear: first year in the daily time series
    param dataout: daily time series array reshaped as described above
    '''
    londimlen  = datain.shape[0]
    latdimlen  = datain.shape[1]
    datain = leap_remove_gridded(datain,startyear,2)
    timedimlen = datain.shape[2]
    extra_date = timedimlen % 365
    # add pseudo values to make the reshape work 
    # (i.e. add enough hours to make it an exact number of years worth of hours)
    sudovals = np.nan * np.ones((londimlen, latdimlen, (365 - extra_date)))
    datain = np.concatenate((datain,sudovals),axis=2)
    newtdim=int(datain.shape[2]//365)
    dataout = np.reshape(datain, (londimlen, latdimlen, newtdim, 365)).transpose((0,1,3,2))
    return dataout

def make_two_year_array(datain):
    tmp1=np.append(datain[:,:,:,0:-1],datain[:,:,:,1:],axis=2)
    sudovals = np.nan * np.ones((datain.shape[0], datain.shape[1],365))
    sudovals=np.expand_dims(sudovals,3)
    lastyear=datain[:,:,:,datain.shape[3]-1]
    lastyear=np.expand_dims(lastyear,3)
    lastyear = np.append(lastyear,sudovals,axis=2)
    dataout = np.append(tmp1,lastyear,axis=3)
    return(dataout)

def leap_remove_gridded(timeseries, datastartyear, timedim):
    """
    This function removes leap days from a time series 
    param timeseries: array containing daily time series
    param datastartyear: start year of the input data
    param timedim: time dimension location
    output data: time series with the leap days removed. 
    """
    data = timeseries
    leaplist=[]
    # system only takes 365 days in each year so we
    # remove leap year values from the long term time series
    if datastartyear % 4 == 1:  # if the start year is not a leap year (Matthew)
        for t in range(1154, data.shape[timedim], 1459):
            leaplist.append(t)
    elif datastartyear % 4 == 2:  # if the start year is not a leap year (Mark)
        for t in range(789, data.shape[timedim], 1459):
            leaplist.append(t)
    elif datastartyear % 4 == 3:  # if the start year is not a leap year (Luke)
        for t in range(424, data.shape[timedim], 1459):
            leaplist.append(t)
    elif datastartyear % 4 == 0:  # if the start year is a leap year (John)
        for t in range(59, data.shape[timedim], 1459):
            leaplist.append(t)
    data=np.delete(data,leaplist,axis=timedim)
    return data

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict) 


#------------------------------------------------------------------------------
# Auto-run
#------------------------------------------------------------------------------
if __name__ == '__main__':
    forecast_wrapper()