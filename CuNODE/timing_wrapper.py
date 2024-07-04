# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 09:21:49 2024

@author: cca79
"""

from time import time
import numpy as np
from functools import wraps

def timing(f, nruns=3):
    @wraps(f)
    def wrap(*args, **kw):
        ts = np.zeros(nruns)
        te = np.zeros(nruns)
        for  i in range(nruns):
            ts[i] = time()
            result = f(*args, **kw)
            te[i] = time()
        durations = te-ts
        print('func:%r took: \n %2.6f sec avg \n %2.6f max \n %2.6f min \n over %d runs' % \
          (f.__name__, np.mean(durations), np.amax(durations), np.amin(durations), nruns))
        return result
    return wrap