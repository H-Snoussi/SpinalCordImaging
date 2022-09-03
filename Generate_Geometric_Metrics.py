#!/usr/bin/env python


# Haykel Snoussi (dr.haykel.snoussi@gmail.com), University of Rennes 1, April 2019 (c)

# 
# Author: Haykel Snoussi (dr.haykel.snoussi@gmail.com)
# Project: EMISEP Study for Spinal Cord Imaging 
# August 2018, Inria Rennes - Bretagne Atlantique and University of Rennes 1, France
#


import argparse

import os, sys
import math
import numpy as np
import nibabel as nib
import csv

from scipy import spatial

import dipy.reconst.dti as dti
from dipy.align.imaffine import get_direction_and_spacings

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import axes3d, art3d

description = "1. Computes Mean angle direction (MAD) and Angulaer concentration directions (ACD), geometric metrics, between spinal cord centerline and model, save results in CSV files, 2. Plot results, 3. Save fitted centerline in png file"
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

#============================fit the centerline=======================
print "Fit the centerline..."
ts = np.linspace(0, 1, nb_points)
if args.fit_method == "polynomial":
    # We now fit a polynomial in t for each of x, y, z for t in [0, 1]
    # We use least-squares fitting, which means we have to compute the pseudoinverse
    degree = 4
    H = np.zeros((nb_points, degree + 1))
    for d in range(degree + 1):
        H[:, d] = ts ** d  # The "observation" matrix has monomials in each column
    Hpinv = np.linalg.pinv(H) #Compute the pseudo-inverse of a matrix
    coefs = np.dot(Hpinv, points_centerline)
    
    # Prepare the Frenet frame computation
    H1p = np.zeros((nb_points, degree + 1))
    for d in range(1, degree + 1):
        H1p[:, d] = d * ts**(d - 1)
    
    H2p = np.zeros((nb_points, degree + 1))
    for d in range(2, degree + 1):
        H2p[:, d] = d * (d - 1) * ts**(d - 2)
    
    centerline = np.dot(H, coefs)      # P
    centerline_1p = np.dot(H1p, coefs) # P'
    centerline_2p = np.dot(H2p, coefs) # P''
elif args.fit_method == "spline":
    from scipy.interpolate import UnivariateSpline
    spline_degree = 3
    smoothing = float(args.smoothing_value)
    spline_x = UnivariateSpline(ts, points_centerline[:, 0], 
                                k=spline_degree, s=smoothing)
    spline_y = UnivariateSpline(ts, points_centerline[:, 1], 
                                k=spline_degree, s=smoothing)
    spline_z = UnivariateSpline(ts, points_centerline[:, 2], 
                                k=spline_degree, s=smoothing)
    centerline = np.array([spline_x(ts), spline_y(ts), spline_z(ts)]).T       # P
    centerline_1p = np.array([spline_x(ts, 1), spline_y(ts, 1), spline_z(ts, 1)]).T  # P'
    centerline_2p = np.array([spline_x(ts, 2), spline_y(ts, 2), spline_z(ts, 2)]).T  # P''
else:
    raise ValueError("fit_method not supported.")



#========================Compute the Frenet frame ==================
print "Compute the Frenet Frame"
speed = np.linalg.norm(centerline_1p, axis=1) # v(t)=norm of P'
acceleration = np.sum(centerline_1p * centerline_2p, axis=1) / speed # v'(t)
tangent_centerline = centerline_1p / speed[:, np.newaxis]
normal_centerline = centerline_2p - \
    acceleration[:, np.newaxis] * tangent_centerline
normal_centerline /= np.linalg.norm(normal_centerline, axis=1)[:, np.newaxis]# ?
binormal_centerline = np.cross(tangent_centerline, normal_centerline, 
                               axisa=1, axisb=1)
#====================================================================





#=======================Nearest_point========================
indices_mask = np.transpose(np.nonzero(mask))
points_mask = np.dot(indices_mask, M.T) + abc 
kd_tree = spatial.KDTree(centerline)
_, indices_nearest_neighbor = kd_tree.query(points_mask)#(1090,)




#====Open tensor image and extract principal eigenvectors===============
print "Extract principal eigenvectors..."
if args.tensor is not None:
  print "Using tensor file, compute e1..."
  prefix = "BallStick"
  tensor = nib.load(args.tensor).get_data()
  # We need to check whether tensor is stored as a 4D image or a 3D image with 6 components.
  if len(tensor.shape) == 5:
    tensor = tensor[..., 0, :]
  evals, evecs = dti.decompose_tensor(dti.from_lower_triangular(tensor[mask]))
  if args.invert_x:
      evecs[..., 0, :] *= -1
  if args.invert_y:
      evecs[..., 1, :] *= -1
  if args.invert_z:
      evecs[..., 2, :] *= -1
  rotation = get_direction_and_spacings(affine_mask, 4)[0]
  if args.no_realign:
    e1 = evecs[..., 0]
  else:
    e1 = np.dot(rotation, evecs[..., 0].T).T   
else: #if there is no tensor files, computes DTI
  print "----> No tensor file given"
  prefix="DTI"
  from dipy.io import read_bvals_bvecs
  bvals, bvecs = read_bvals_bvecs(args.bvals, args.bvecs)
  from dipy.core.gradients import gradient_table
  gtab = gradient_table(bvals, bvecs)
  print 'Computing DTI using dipy...'
  import dipy.reconst.dti as dti
  tenmodel = dti.TensorModel(gtab)
  tenfit = tenmodel.fit(data_dmri[mask])
  evecs = tenfit.evecs
  if args.invert_x:
      evecs[..., 0, :] *= -1
  if args.invert_y:
      evecs[..., 1, :] *= -1
  if args.invert_z:
      evecs[..., 2, :] *= -1
  print "Compute e1"
  from dipy.align.imaffine import get_direction_and_spacings
  rotation = get_direction_and_spacings(affine_dmri, 4)[0]
  if args.no_realign:
    e1 = evecs[..., 0]
  else:
    e1 = np.dot(rotation, evecs[..., 0].T).T   
  print e1.shape




  # #====Open tensor image and extract principal eigenvectors===============
  # tensor = nib.load(args.tensor).get_data()
  # # We need to check whether tensor is stored as a 4D image or a 3D image with 
  # # 6 components.
  # if len(tensor.shape) == 5:
  #     tensor = tensor[..., 0, :]
  # evals, evecs = dti.decompose_tensor(dti.from_lower_triangular(tensor[mask]))
  # rotation = get_direction_and_spacings(affine_mask, 4)[0]
  # if args.no_realign:
  #     e1 = evecs[..., 0]
  # else:
  #     e1 = np.dot(rotation, evecs[..., 0].T).T


#==Place principal eigenvector into the Frenet frame everywhere====
# nb_points = np.count_nonzero(mask)#nber of nonzero pts in mask
nb_points_mask=indices_mask.shape[0]
e1_frenet = np.zeros((nb_points_mask, 3))#(1090,3)
for i in range(nb_points_mask):
    j = indices_nearest_neighbor[i]
    frenet_frame = np.vstack((tangent_centerline[j],
                              normal_centerline[j],
                              binormal_centerline[j])).T
    e1_frenet[i] = np.dot(frenet_frame.T, e1[i])

self_products_e1 = np.zeros((nb_points_mask, 3, 3))
for i in range(nb_points_mask):
   #The next line actually computes e1 * e1.T to get a 3x3 matrix
   self_products_e1[i] = np.outer(e1_frenet[i], e1_frenet[i]) 

# print e1[nb_points_mask // 2]
# print e1_frenet[nb_points_mask // 2]
# print self_products_e1[nb_points_mask // 2]

#===Compute per-level "mean" orientation========= 
labels = nib.load(args.labels).get_data()[mask > 0]#(1090,) retunrs indices
label_ids = np.unique(labels[labels > 0])#Returns the sorted unique elements of an array.

actual_path=args.output_path

# output_file_txt_evals = actual_path + "_mean_direction_tensor_evals_dmri_" + prefix + ".txt"
# output_file_txt_angle = actual_path + "_mean_angle_dmri_" + prefix + ".txt"




output_file_csv_evals = actual_path + "_mean_direction_tensor_evals_dmri_" + prefix + ".csv"
output_file_csv_angle = actual_path + "_mean_angle_dmri_" + prefix + ".csv"

print "Create csv files for mean angle and mean direction tensor"

if not os.path.isfile(output_file_csv_evals):
  with open(output_file_csv_evals, 'w') as csvfile:
    fieldnames = ['subject_id', 'vertebral_level', 'metric_value']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

if not os.path.isfile(output_file_csv_angle):
  with open(output_file_csv_angle, 'w') as csvfile:
    fieldnames = ['subject_id', 'vertebral_level', 'metric_value']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

from collections import Counter
#for label_id in label_ids:
mean_direction_tensors = np.zeros((11, 3, 3))
# """
labels_of_interest = range(1, 10)
# mask_label = (labels == 0)
for label_id in labels_of_interest:
  mask_label = (labels == label_id)
  # mask_label = np.logical_or(mask_label, labels==label_id)
  # print "mask_label", Counter(mask_label)
  # """
  # for label_id in range(1,11):
  #     mask_label = (labels == label_id)
  #     # label_id=17
  mean_direction_tensor = np.mean(self_products_e1[mask_label], axis=0)
  #mean_direction_tensors[label_id - 1] = mean_direction_tensor
  # linalg.eigh : Return the eigenvalues and eigenvectors of a Hermitian or symmetric matrix
  mean_direction_tensor_evals, mean_direction_tensor_evecs = \
      np.linalg.eigh(mean_direction_tensor)

  # Trigonometric inverse cosine, element-wise.
  mean_angle = np.arccos(np.abs(mean_direction_tensor_evecs[0, 2]))
  # print("label: ", label_id)
  #print("principal eigenvector: ", mean_direction_tensor_evecs[:, 2])
  #print("principal eval =", mean_direction_tensor_evals[2])
  # print(mean_direction_tensor_evals[2])
  # print("mean angle: ", np.rad2deg(mean_angle))
  # outfile_evals=open(output_file_txt_evals,"a")
  # outfile_evals.write(str(mean_direction_tensor_evals[2])+ '\n')
  # outfile_angle=open(output_file_txt_angle,"a")
  # outfile_angle.write(str(np.rad2deg(mean_angle))+ '\n')

  with open(output_file_csv_evals, 'a') as csvfile:
      fieldnames = ['subject_id', 'vertebral_level', 'metric_value']
      writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
      writer.writerows([{'subject_id':"Test" 'vertebral_level': "C"+str(label_id), 'metric_value': str(mean_direction_tensor_evals[2])}])
  with open(output_file_csv_angle, 'a') as csvfile:
      fieldnames = ['subject_id', 'vertebral_level', 'metric_value']
      writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
      writer.writerows([{'subject_id':"Test" 'vertebral_level': "C"+str(label_id), 'metric_value': str(np.rad2deg(mean_angle))}])


#======================== Compute mask of (uniquely) the centerline ===========
# we find indices of nearest neighbour in the mask
indices = np.zeros(nb_points, dtype=np.int)
for id_point in range(nb_points):
    indices[id_point] = np.argmin(np.sum((points_mask - centerline[id_point][np.newaxis, :])**2, axis=1))
# print indices

my_colors = ["k", "r", "g", "b", "c", "m", "y", "r", "g", "b", "c", "m", "y", "r", "g", "b", "c", "m", "y"]
labels = np.array(labels, dtype=np.int)
display_colors = [my_colors[l] for l in labels[indices]]

print "Prepare for plotting per-level mean direction tensor"
#===================== Prepare for plotting per-level mean direction tensor =======================
from scipy import ndimage

# centers of labels
labels_array = np.array(nib.load(args.labels).get_data())
center_labels = np.zeros((7,3))

for label_id in range(1,8):
  center_labels[label_id-1,:] = ndimage.measurements.center_of_mass(mask, labels_array, label_id)
points_center_labels = np.dot(center_labels, M.T) + abc 

kd_tree_labels = spatial.KDTree(centerline)
_, indices_nearest_center_labels = kd_tree.query(points_center_labels)#(1090,)
centers_ellipses = centerline[indices_nearest_center_labels,:]

# mean_direction_tensor in Frenet frame, and after projecting into xz and yz planes
angles_zx = np.zeros(7)
lambdas_zx = np.zeros((7, 2))
angles_zy = np.zeros(7)
lambdas_zy = np.zeros((7, 2))
for i in range(7):
    j = indices_nearest_center_labels[i]
    frenet_frame = np.vstack((tangent_centerline[j],
                              normal_centerline[j],
                              binormal_centerline[j])).T
    mean_direction_tensor = np.dot(np.dot(frenet_frame, mean_direction_tensors[i]), frenet_frame.T)
    Pzx = np.array(([[0, 0, 1], [1, 0, 0]]))
    tensor_zx = np.dot(np.dot(Pzx, mean_direction_tensor), Pzx.T)
    tensor_zx_evals, tensor_zx_evecs = np.linalg.eigh(tensor_zx)
    angles_zx[i] = np.rad2deg(np.arctan2(tensor_zx_evecs[1, 1], tensor_zx_evecs[0, 1]))
    lambdas_zx[i] = tensor_zx_evals

    Pzy = np.array(([[0, 0, 1], [0, 1, 0]]))
    tensor_zy = np.dot(np.dot(Pzy, mean_direction_tensor), Pzy.T)
    tensor_zy_evals, tensor_zy_evecs = np.linalg.eigh(tensor_zy)
    angles_zy[i] = np.rad2deg(np.arctan2(tensor_zy_evecs[1, 1], tensor_zy_evecs[0, 1]))
    lambdas_zy[i] = tensor_zy_evals
#===================================================================


Plot_fitted_centerline=0

if Plot_fitted_centerline:
  #============================ Plot fitted centerline ===============
  ts = np.linspace(0, 1, 100)
  padding = 5
  zoom = 20
  fig = plt.figure(figsize=(13, 17))

  ax2 = fig.add_subplot(2, 1, 1, aspect="equal")
  # plt.scatter(points_centerline[:, 2], points_centerline[:, 0], 
  #     label="data points")
  # plt.scatter(centers_ellipses[:, 2], centers_ellipses[:, 0], 
  #     label="Vertebral level centers")
  plt.plot(centerline[:, 2], centerline[:, 0], 
      label="%s fit" % args.fit_method)
  # ax2.quiver(centerline[:, 2], centerline[:, 0],
  #           e1[indices, 2], e1[indices, 0], units="xy", color=display_colors, width=0.1)
  for i in range(7):
      e = Ellipse(xy=centers_ellipses[i, [2, 0]], angle=angles_zx[i], 
                  width=zoom * lambdas_zx[i, 1], height=zoom * lambdas_zx[i, 0])
      e.set_color("r")
      ax2.add_artist(e)
  # plt.plot(points_centerline[:, 2], spline_x(points_centerline[:, 2]), 
  #     label="spline fit")
  # plt.plot(z_centerline_fit, x_centerline_fit, label="NURBS fit")
  plt.ylim(np.min(points_centerline[:, 0]) - padding, np.max(points_centerline[:, 0]) + padding)
  # plt.legend()


  ax3 = fig.add_subplot(2, 1, 2, aspect="equal")
  # plt.scatter(points_centerline[:, 2], points_centerline[:, 1], 
  #     label="data points")
  # plt.scatter(centers_ellipses[:, 2], centers_ellipses[:, 1], 
  #     label="Vertebral level centers")
  plt.plot(centerline[:, 2], centerline[:, 1], 
      label="%s fit" % args.fit_method)
  # ax3.quiver(centerline[:, 2], centerline[:, 1], 
  #            e1[indices, 2], e1[indices, 1], units="xy", color=display_colors, width=0.1)
  for i in range(7):
      e = Ellipse(xy=centers_ellipses[i, [2, 1]], angle=angles_zy[i], 
                  width=zoom * lambdas_zy[i, 1], height=zoom * lambdas_zy[i, 0])
      e.set_color("r")
      ax3.add_artist(e)
  # plt.plot(points_centerline[:, 2], spline_y(points_centerline[:, 2]), 
  #     label="spline fit")
  # plt.plot(z_centerline_fit, y_centerline_fit, label="NURBS fit")
  plt.ylim(np.min(points_centerline[:, 1]) - padding, np.max(points_centerline[:, 1]) + padding)
  # plt.legend()
   #plt.show()
  #===================================================================

Compute_the_centerline_points=0
if Compute_the_centerline_points:
  #================Compute the centerline points=======================
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


Plot_restults=0
if Plot_restults:
  # =======================We plot the result===========================
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d', aspect='equal')
  center = np.sum(centerline, axis=0) / nb_points
  ax.scatter(points_centerline[:, 0] - center[0], 
             points_centerline[:, 1] - center[1], 
             points_centerline[:, 2] - center[2], color='r')
  ax.plot(centerline[:, 0] - center[0],
          centerline[:, 1] - center[1],
          centerline[:, 2] - center[2], color='k')
  maxval = np.max(np.abs(points_centerline))
  ax.set_xlim3d(-maxval, maxval)
  ax.set_ylim3d(-maxval, maxval)
  ax.set_zlim3d(-maxval, maxval)
  ax.quiver(centerline[:, 0] - center[0], 
            centerline[:, 1] - center[1], 
            centerline[:, 2] - center[2],
            tangent_centerline[:, 0], 
            tangent_centerline[:, 1], 
            tangent_centerline[:, 2], length=1.0)
  ax.quiver(centerline[:, 0] - center[0], 
            centerline[:, 1] - center[1], 
            centerline[:, 2] - center[2],
            normal_centerline[:, 0], 
            normal_centerline[:, 1], 
            normal_centerline[:, 2], length=1.0)
  ax.quiver(centerline[:, 0] - center[0], 
            centerline[:, 1] - center[1], 
            centerline[:, 2] - center[2],
            binormal_centerline[:, 0], 
            binormal_centerline[:, 1], 
            binormal_centerline[:, 2], length=1.0)
  ax.quiver(points_mask[:, 0] - center[0], 
            points_mask[:, 1] - center[1], 
            points_mask[:, 2] - center[2],
            e1[:, 0], 
            e1[:, 1], 
            e1[:, 2], length=1.0)

  plt.show()

"""

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', aspect='equal')
center = np.sum(centerline, axis=0) / nb_points
ax.scatter(points_centerline[:, 0] - center[0], 
           points_centerline[:, 1] - center[1], 
           points_centerline[:, 2] - center[2], color='red',label='Barycenters')
ax.plot(centerline[:, 0] - center[0],
        centerline[:, 1] - center[1],
        centerline[:, 2] - center[2], color='grey',label='Fitted centerline')
maxval = np.max(np.abs(points_centerline))
ax.set_xlim3d(-maxval, maxval)
ax.set_ylim3d(-maxval, maxval)
ax.set_zlim3d(-maxval, maxval)
ax.quiver(centerline[:, 0] - center[0], 
          centerline[:, 1] - center[1], 
          centerline[:, 2] - center[2],
          tangent_centerline[:, 0], 
          tangent_centerline[:, 1], 
          tangent_centerline[:, 2], length=1.50,color='blue',label='Tangent vector')
ax.quiver(centerline[:, 0] - center[0], 
          centerline[:, 1] - center[1], 
          centerline[:, 2] - center[2],
          normal_centerline[:, 0], 
          normal_centerline[:, 1], 
          normal_centerline[:, 2], length=1.50,color='green',label='Normal vector')
ax.quiver(centerline[:, 0] - center[0], 
          centerline[:, 1] - center[1], 
          centerline[:, 2] - center[2],
          binormal_centerline[:, 0], 
          binormal_centerline[:, 1], 
          binormal_centerline[:, 2], length=1.50,color='red',label='Binomal vector')
# ax.quiver(points_mask[:, 0] - center[0], 
#           points_mask[:, 1] - center[1], 
#           points_mask[:, 2] - center[2],
#           e1[:, 0], 
#           e1[:, 1], 
#           e1[:, 2], length=1.0)
plt.legend(loc='best', shadow=False, scatterpoints=1)
# plt.show()

# savefig('foo.pdf', bbox_inches='tight')
