__all__ = [s for s in dir() if not s.startswith('_')]

import os.path
from . import server
# from . import attenuation
from . import io
# from . import psd
# from . import env
from . import GPM
from . import misc
from . import vis
# from . import refractive
from . import simulate
from . import retrieve
from . import representation

import matplotlib.pyplot as plt
import matplotlib.style as style



plt.ion()
# plt default plot properties
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('image', cmap='plasma')

style.use('tableau-colorblind10')


module_path = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
