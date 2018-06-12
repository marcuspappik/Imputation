# -*- coding: utf-8 -*-
# @Author: Marcus Pappik
# @Date:   2018-06-12 17:16:53
# @Last Modified by:   marcus
# @Last Modified time: 2018-06-12 17:38:32

import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale


def L2_score(original, result):
    diff = minmax_scale(original - result)
    return np.sqrt(np.sum(np.square(diff)))


def L1_score(original, result):
    diff = minmax_scale(original - result)
    return np.sum(np.absolute(diff))
