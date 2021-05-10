# testing the tracer module

import tracer
import numpy as np
import matplotlib.pyplot as plt

data_file = '/home/anderson/Documents/UW_AM_Research/Data/plasma_sep2.txt'
data, names = tracer.getdata(tracer.make_file(data_file))

# get border and triangles and check plots
border, triangles = tracer.alpha_shape(data[:,:2])
tracer.geoplot(data, data.shape[0], [0,1], (12,9), border=border)
plt.show(block=True)
