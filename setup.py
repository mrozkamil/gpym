from setuptools import setup, find_packages

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='gpym',
      version='0.1.0',
      description='tools to analyse the GPM-DPR data',
      url='https://github.com/mrozkamil/gpym',
      author='Kamil Mroz',
      author_email='kamil.mroz@le.ac.uk',
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      include_package_data=True,
      install_requires=[
          'numpy', 'scipy', 
          'matplotlib', 'cartopy', 'seaborn',
          'xarray', 'dask', 'h5py', 'netCDF4', 'bottleneck', 'h5netcdf', 
          'astropy', 'itur', 'dill',])
