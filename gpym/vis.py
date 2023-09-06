import matplotlib.pyplot as plt
import seaborn as sns
from xhistogram.xarray import histogram as xhist
from scipy.ndimage.filters import convolve
import numpy as np

def inpolygon(xq, yq, xv, yv):
    shape = xq.shape
    xq = xq.ravel()
    yq = yq.ravel()
    xv = xv.ravel()
    yv = yv.ravel()
    q = [(xq[i], yq[i]) for i in range(xq.size)]
    p = path.Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
    xq.reshape(shape)
    yq.reshape(shape)
    return p.contains_points(q).reshape(shape)

def add_rectangle(xy,width,height, color='k', alpha=0.2):
    rect=plt.Rectangle(xy, width, height, color=color, alpha=alpha,
                   edgecolor=None,linewidth=0,joinstyle='miter')
    plt.gca().add_patch(rect)

def gaussian2d(x,y,mean=(0,0), std=(1,1)):

    if x.shape == y.shape and len(x.shape)==2:
        X,Y = x,y
    else:
        X,Y=np.meshgrid(x,y)
    ker=np.exp(-((X-mean[0])/std[0])**2-((Y-mean[1])/std[1])**2)
    return ker/sum(ker.ravel())

def lon_lat_plot(lon_data,lat_data, var_data, lon0,lat0, label = '$Z^{S}$ [dBZ]',
                 vmin = 15., vmax = 50., cmap = 'plasma', scatter = False,
                 ran_deg = 2., marker_size = 8):

    lon_min, lon_max = lon0-ran_deg, lon0+ran_deg
    lat_min, lat_max = lat0-ran_deg, lat0+ran_deg

    circle_points_100km = geodesic.Geodesic().circle(
                lon=lon0, lat=lat0, radius=1e5, n_samples=1000, endpoint=False)
    # circle_points_200km = geodesic.Geodesic().circle(
    #     lon=lon0, lat=lat0, radius=2e5, n_samples=1000, endpoint=False)


    fig = plt.figure(figsize=(6.0, 5.))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max,], crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(lon_min//2*2+2,lon_max,2.), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(lat_min//2*2+2,lat_max,2.), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    if scatter:
        ind_good = np.where(np.isfinite(var_data))
        pc = ax.scatter(lon_data[ind_good],lat_data[ind_good], c = var_data[ind_good], s = marker_size,
                         vmin = vmin, vmax = vmax, cmap = cmap)
    else:
        pc = ax.pcolor(lon_data,lat_data, var_data, vmin = vmin, vmax = vmax, cmap = cmap)
    plt.colorbar(pc, pad = 0.02, aspect = 30, label = label, fraction = 0.1)
    # ax.text(x = gv_ds.site_lon.data.item(), y = gv_ds.site_lat.data.item(), s = 'GR', transform=ccrs.PlateCarree(),
    #         fontsize =16,)
    ax.plot(gv_ds.site_lon.data.item(), gv_ds.site_lat.data.item(), 'xk')
    ax.set_title('')
    ax.plot(circle_points_100km[:,0], circle_points_100km[:,1], '--k', lw =1, label = '100 km radius', transform=ccrs.PlateCarree())
    # ax.plot(circle_points_200km[:,0], circle_points_200km[:,1], ':k', lw =1, label = '200 km radius', transform=ccrs.PlateCarree())
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.coastlines()
    gl = ax.gridlines()
    gl.right_labels = False
    gl.top_labels = False
    gl.xlocator = mticker.FixedLocator(np.arange(lon_min//1+1,lon_max,1.))
    gl.ylocator = mticker.FixedLocator(np.arange(lat_min//1+1,lat_max,1.))
    plt.tight_layout()

    return fig, ax




def kdeplot(x, y, cmap = 'plasma', shade = True, thresh=0.01, cbar_format = '%.3f',
        xlabel= None, ylabel = None, levels=10 ):


    fig, ax =plt.subplots(figsize=(8, 6))
    ax = sns.kdeplot(x=x, y=y, cmap=cmap, shade=shade, thresh=thresh,
            ax = ax, cbar = True, levels = levels,
            cbar_kws = {'pad': 0.02, 'aspect':30, 'format': cbar_format, 'fraction': 0.1}, )
    ax.grid()
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    plt.tight_layout()
    return fig, ax



def densityplot(x, y, cmap = 'plasma', thresh=0.005, cbar_format = '%.3f',
        xlabel= None, ylabel = None, levels=10, bins = 151, ker_size = 7, ):

    if isinstance(bins,int):
        bins_x = np.linspace(x.min(), x.max(), bins)
        bins_y = np.linspace(y.min(), y.max(), bins)
        bins = (bins_x, bins_y)
    if isinstance(bins,list) or isinstance(bins, tuple):
        if len(bins)==1:
            bins = (bins[0], bins[0])

    tmp_hist = xhist( x, y, bins = bins,)
    tmp_area = np.diff(bins[0])[:, np.newaxis] * np.diff(bins[1])

    # tmp_pdf = xhist( x, y, bins = bins, density=True)
    # tmp_pdf2 = tmp_hist/tmp_hist.sum()/tmp_area

    #smoothing
    kernel=gaussian2d(x=np.linspace(-3,3,ker_size),y=np.linspace(-3,3,ker_size),
                mean=(0,0), std=(1,1))

    smooth_pdf = convolve(tmp_hist,kernel, mode = 'constant')
    tmp_area = convolve(tmp_area,kernel, mode = 'constant')


    smooth_pdf = smooth_pdf/tmp_area/smooth_pdf.sum()
    smooth_pdf_sum =  smooth_pdf.sum()
    smooth_pdf_unique = np.sort(smooth_pdf.ravel())
    smooth_cdf_unique = np.cumsum(smooth_pdf_unique)
    ind_thr = np.argmax(smooth_cdf_unique>smooth_pdf_sum*thresh)


    smooth_pdf = np.ma.masked_where(smooth_pdf<smooth_pdf_unique[ind_thr], smooth_pdf)

    binc = [0.5*(bins[ii][1:] + bins[ii][:-1]) for ii in range(2)]

    fig, ax =plt.subplots(figsize=(8, 6))
    im=ax.contourf(binc[0],binc[1],smooth_pdf.T, levels = levels,
                    cmap=cmap, extend='neither', corner_mask=False)
    cbar = plt.colorbar(im, pad=0.02, aspect=30,
                    format = cbar_format, fraction = 0.1)
    ax.grid()
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    plt.tight_layout()

    return fig, ax, tmp_hist, smooth_pdf, binc


