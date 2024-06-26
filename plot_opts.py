# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 17:38:25 2024

@author: cca79
"""

import numpy as np
import matplotlib.pyplot as plt

opt_timings = np.loadtxt("ysize_timings")

ysize = np.arange(58) + 4

plt.plot(ysize, opt_timings.T)