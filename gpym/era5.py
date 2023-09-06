from scipy.ndimage import map_coordinates
import numpy as np
import xarray as xr
# import time
import warnings
warnings.filterwarnings('ignore')
import datetime as dt
import os

class era5():
    def __init__(self, era5_dir = '/badc/ecmwf-era5/data/oper/an_sfc/%04d/%02d/%02d',
                    era5t_dir = '/badc/ecmwf-era5t/data/oper/an_sfc/%04d/%02d/%02d'):
        self.era5_dir = era5_dir
        self.era5t_dir = era5t_dir


        #key is the variable name, value is the file name convention
        self.era5_vars = {  'd2m':    '2d',     # 2 metre dewpoint temperature
                            't2m':    '2t',     # 2 metre temperature
                            'skt':    'skt',    # skin temperature (including see)
                            'tcwv':   'tcwv',   # total column water vapour
                            'sd':     'sd',     # snow depth
                            'asn':    'asn',    # snow albedo
                            'siconc': 'ci',     # Sea ice area fraction
                            'u10':    '10u',    # 10 metre U wind component
                            'v10':    '10v',    # 10 metre V wind component
                            'cape':   'cape',   # Convective available potential energy
                            'msl':    'msl',    # Mean sea level pressure
                            'sst':    'sst',    # Sea surface temperature
                            'tcc':    'tcc',    # Total cloud cover
                            }

    def get_coord(self, xi, x, extrap = False):
        """this is a method to estimate coordinates of query points
            with respect to the coordinate grid points """

        tmp_dx = np.unique(np.diff(x))
        #check if the input grid is regular
        is_regular = (tmp_dx.size ==1)
        #check if the input points are in the ascending order
        is_ascending = (x[0]<x[-1])

        if is_regular:
            if is_ascending:
                c = (xi-x[0])/(x[-1]-x[0])*(x.size-1)
            else:
                c = (x.size-1) - (xi-x[-1])/(x[0]-x[-1])*(x.size-1)
            if not extrap:
                c[c<0] = 0
                c[c>(x.size-1)] = (x.size-1)

        else:
            if is_ascending:
                c = np.interp(xi, x, np.arange(x.size))
                if extrap:
                    ind_extr = np.where(xi<x[0])
                    dx = x[1]-x[0]
                    c_extrap = (xi[ind_extr]-x[0])/dx
                    c[ind_extr] = c_extrap

                    ind_extr = np.where(xi>x[-1])
                    dx = x[-1]-x[-2]
                    c_extrap = (xi[ind_extr]-x[-1])/dx
                    c[ind_extr] = c_extrap


            else:
                ind_sort = np.arange(x.size)[::-1]
                c = np.interp(xi, x[::-1], ind_sort)
                if extrap:
                    ind_extr = np.where(xi>x[0])
                    dx = x[1]-x[0]
                    c_extrap = (xi[ind_extr]-x[0])/dx
                    c[ind_extr] = c_extrap

                    ind_extr = np.where(xi<x[-1])
                    dx = x[-1]-x[-2]
                    c_extrap = (xi[ind_extr]-x[-1])/dx
                    c[ind_extr] = c_extrap
        return c

    def get_time_stamps(self, time_da):
        time_s = time_da.min().values.astype('datetime64[ms]').astype(
            dt.datetime).replace(microsecond=0, second=0, minute=0)
        time_e = time_da.max().values.astype('datetime64[ms]').astype(
            dt.datetime).replace(microsecond=0, second=0, minute=0) + (
            dt.timedelta(hours = 1))


        dd = time_s + dt.timedelta(hours = 0)
        era5_time_stamps = [dd,]
        while dd<=time_e:
            dd += dt.timedelta(hours = 1)
            era5_time_stamps.append(dd)
        return era5_time_stamps

    def __call__(self, era5_v, lon_da, lat_da, time_da, interp = 'xarray', **interp_kwargs):
        # now = time.time()
        time_stamps = self.get_time_stamps(time_da)
        era5_files = []
        for dd in time_stamps:
            tmp_dir = self.era5_dir % (dd.year, dd.month, dd.day)
            if not os.path.exists(tmp_dir):
                tmp_dir = self.era5t_dir % (dd.year, dd.month, dd.day)
            if not os.path.exists(tmp_dir):
                return

            tmp_list = [os.path.join(tmp_dir,fn) for fn in os.listdir(tmp_dir)
                if '{:04d}{:02d}{:02d}{:02d}00.{}.nc'.format(
                    dd.year, dd.month, dd.day, dd.hour,
                    self.era5_vars[era5_v]) in fn]
            era5_files += tmp_list

        ds_ecmwf = xr.open_mfdataset(era5_files,
                combine='nested', concat_dim = 'time')

        lon_da = lon_da.where(lon_da>=0, lon_da+360.)

        if interp == 'xarray':

            tmp_da = ds_ecmwf[era5_v].isel(longitude = [0,])
            tmp_da['longitude'] = tmp_da['longitude']+ 360

            da_interp = xr.combine_nested([ds_ecmwf[era5_v], tmp_da],
                    concat_dim = 'longitude', **interp_kwargs)

            name_lon = lon_da.name
            name_lat = lat_da.name
            name_time = time_da.name

            if name_lon == 'longitude':
                lon_da.name = 'Lon'
            if name_lat == 'latitude':
                lat_da.name = 'Lat'
            if name_time == 'time':
                time_da.name = 'Time'

            da_out = da_interp.interp(longitude = lon_da,
                                    latitude = lat_da,
                                    time = time_da, )
            da_out = da_out.compute()


            da_out = da_out.reset_coords(names="longitude", drop=True)
            da_out = da_out.reset_coords(names="latitude", drop=True)
            da_out = da_out.reset_coords(names="time", drop=True)

            if name_lon == 'longitude':
                da_out = da_out.rename({'Lon': 'longitude'})
            if name_lat == 'latitude':
                da_out = da_out.rename({'Lat': 'latitude'})
            if name_time == 'time':
                da_out = da_out.rename({'Time': 'time'})
            # print('%.1f' % (time.time() - now,))
            return da_out
        if interp == 'map_coordinates':

            da_dims = ds_ecmwf[era5_v].dims
            coords_dict = {'longitude': self.get_coord(
                xi = lon_da.data, x = ds_ecmwf.longitude.data, extrap = False),
                'latitude': self.get_coord(
                xi = lat_da.data, x = ds_ecmwf.latitude.data, extrap = False),
                'time': self.get_coord(
                xi = time_da.data, x = ds_ecmwf.time.data, extrap = False),}

            X = np.array([coords_dict[k] for k in da_dims])

            interp_val = map_coordinates(ds_ecmwf[era5_v].data,
                                         coordinates = X, mode = 'grid-wrap',)

            da_out = xr.DataArray(data = interp_val, dims = lon_da.dims,
                coords = {lon_da.name: lon_da, lat_da.name: lat_da, time_da.name: time_da})
            # print('%.1f' % (time.time() - now,))
            return da_out


