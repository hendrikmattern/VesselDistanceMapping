from vdm import vessel_phantom
import numpy as np

# todo: convert to unittest
# zero test
if not vessel_phantom.cartesian2spherical_in_degree(0, 0, 0) == (0, 0, 0):
    print("fail0")

# x-unit vector -> r=1, phi=0, theta=90
if not vessel_phantom.cartesian2spherical_in_degree(1, 0, 0) == (1, 0, 90):
    print("failx")

# negative y-unit vector -> r=1, phi=-90, theta=90
if not vessel_phantom.cartesian2spherical_in_degree(0, -1, 0) == (1, -90, 90):
    print("faily")

# 10 time negative z-unit vector -> r=10, phi=0, theta=180
if not vessel_phantom.cartesian2spherical_in_degree(0, 0, -10) == (10, 0, 180):
    print("failz")

# (sqrt(1+1+1), 45 degrees, 54.7 degrees  == xyz(1,1,1)
if not vessel_phantom.spherical2cartesian_in_degree(np.sqrt(3), 45, np.degrees(np.arctan2(np.sqrt(2), 1))) == (1, 1, 1):
    print("failxyz")
