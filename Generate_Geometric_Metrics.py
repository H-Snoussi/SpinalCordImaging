#!/usr/bin/env python


# Haykel Snoussi (dr.haykel.snoussi@gmail.com), University of Rennes 1, April 2019 (c)

import argparse
import os, sys
import math
import numpy as np
import nibabel as nib
import csv
import dipy.reconst.dti as dti

from scipy import spatial
from scipy import ndimage
from scipy.interpolate import UnivariateSpline

from dipy.align.imaffine import get_direction_and_spacings
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.align.imaffine import get_direction_and_spacings

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

from mpl_toolkits.mplot3d import axes3d, art3d
from collections import Counter




description = "1. Computes Mean angle direction (MAD) and Angular Concentration directions (ACD). Results are saved in a CSV files. \n2. Plot results. \n3. Save fitted centerline in png file."
parser = argparse.ArgumentParser(description=description)
parser.add_argument("-m", dest="mask",
                          required=True,
                          help="Input mask image.")
parser.add_argument("-l", dest="labels",
                          required=True,
                          help="Input label image (expected vertebral levels).")
parser.add_argument("-d", dest="input_dmri",
                          required=True,
                          help="diffusion MRI image.")
parser.add_argument("-e", dest="bvecs",
                          required=True,
                          help="bvec matrix.")
parser.add_argument("-a", dest="bvals",
                          required=True,
                          help="bval matrix.")
parser.add_argument("-t", dest="tensor",
                          required=False,
                          help="Tensor image. If it's not given, the script compute DTI using DIPY")
parser.add_argument("-o", dest="output_path",
                          required=True,
                          help="path of output text file")
parser.add_argument("--no_realign", action="store_true",
                          help="If tensor is already in real-world coordinates, this option tells the script not to apply rotation to the tensors.")
parser.add_argument("--remove_outliers", action="store_true",
                          help="Remove outliers prior to curve fitting for the centerline.")
parser.add_argument("-f", dest="fit_method", default="spline", choices=["spline", "polynomial"],
                          help="Fit method for the centerline (either 'spline' or 'polynomial')")
parser.add_argument("--ix", dest="invert_x", action="store_true",
                          help="Invert x axis for bvecs.")
parser.add_argument("--iy", dest="invert_y", action="store_true",
                          help="Invert y axis for bvecs.")
parser.add_argument("--iz", dest="invert_z", action="store_true",
                          help="Invert z axis for bvecs.")
parser.add_argument("-s", dest="screenshot", default=None,
                          help="filename (png image) for screenshot.")
parser.add_argument("-n", dest="smoothing_value", required=True,
                          help="Positive smoothing factor used to choose the number of knots for Spline fitting method")
args = parser.parse_args()



#===========Get mask image from input file============================
mask_img = nib.load(args.mask)
affine_mask = mask_img.get_affine() # (4, 4)
mask = np.array(mask_img.get_data(), dtype=np.bool)
dim_i, dim_j, dim_k = mask.shape
M = affine_mask[:3, :3] # (3, 3)
abc = affine_mask[:3, 3] # (3,)
#====================================================================




#===========Get dmri image from input file============================
img_dmri = nib.load(args.input_dmri)
affine_dmri = img_dmri.get_affine()
data_dmri = np.array(img_dmri.get_data())






#================Compute the centerline points=======================
print "Compute the centerline points"
points_centerline = []
for j in range(dim_j):
    if np.any(mask[:, j, :]):
        indices_ik = np.transpose(np.nonzero(mask[:, j, :])) #nonzero pts
        nb_points = indices_ik.shape[0] # number of nonzero pts
        indices_ijk = np.vstack((indices_ik[:, 0], 
                             j * np.ones(nb_points), 
                             indices_ik[:, 1])).T #coordinates of nonzero pts
        points = np.dot(M, indices_ijk.T) + abc[:, np.newaxis] # M * indices + abc
        barycenter = np.mean(points, axis=-1)
        points_centerline.append(barycenter) #points_centerline.shape(?, 3)
points_centerline = np.array(points_centerline)
nb_points = points_centerline.shape[0]
#====================================================================





#=================Remove outliers (if option is set)=================
if args.remove_outliers:
  print "Remove outliers..."
  kernel = 3
  interpolated_points = np.zeros((nb_points, 3))
  for k in range(nb_points):
      weights = np.zeros(nb_points)
      for l in range(k - kernel, k + kernel + 1):
          if l >= 0 and l < nb_points and l != k:
              weights[l] = 1.0
      weights /= np.sum(weights)
      interpolated_points[k] = np.sum(weights[:, np.newaxis] * points_centerline,
          axis=0)
  distances = np.sqrt(np.sum((interpolated_points - points_centerline)**2, 
    axis=1))
  indices_not_outlier = (distances < np.mean(distances) + 3 * np.std(distances))
  points_centerline = points_centerline[indices_not_outlier]
  nb_points = points_centerline.shape[0]
#====================================================================
