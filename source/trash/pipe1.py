import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split

from external_master import single_img_features
from external_master import visualize
from external_master import extract_features

basedir = "vehicles/"
image_types = os.listdir(basedir)
cars = []
for imtype in image_types:
	cars.extend(glob.glob(basedir+imtype+'/*'))
print("Number of Vehicle Images found", len(cars))


basedir = "non-vehicles/"
image_types = os.listdir(basedir)
notcars = []
for imtype in image_types:
	notcars.extend(glob.glob(basedir+imtype+'/*'))
print("Number of NotVehicle Images found", len(notcars))


car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0,len(notcars))

car_image = mpimg.imread(cars[car_ind])
notcar_image = mpimg.imread(notcars[notcar_ind])

color_space = "RGB"
orient = 6
pix_per_cell = 8
cell_per_block = 2
hog_channel = 0
spatial_size = (16,16)
hist_bins = 16
spatial_feat = True
hist_feat = True
hog_feat = True



car_features, car_hog_image = single_img_features(car_image, 
	color_space=color_space,
	spatial_size=spatial_size,
	hist_bins=hist_bins,
	orient=orient,
	pix_per_cell=pix_per_cell,
	cell_per_block=cell_per_block,
	hog_channel=hog_channel,
	spatial_feat=spatial_feat,
	hist_feat=hist_feat,
	hog_feat=hog_feat,
	vis=True)

notcar_features, notcar_hog_image = single_img_features(notcar_image, 
	color_space=color_space,
	spatial_size=spatial_size,
	hist_bins=hist_bins,
	orient=orient,
	pix_per_cell=pix_per_cell,
	cell_per_block=cell_per_block,
	hog_channel=hog_channel,
	spatial_feat=spatial_feat,
	hist_feat=hist_feat,
	hog_feat=hog_feat,
	vis=True)


images = [car_image, car_hog_image, notcar_image, notcar_hog_image]
titles = ["","","",""]
fig = plt.figure(figsize=(12,3))
visualize(fig, 1, 4, images, titles)



color_space = "YCrCb"
#color_space = "RGB"
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL"
#hog_channel = 0
#spatial_size = (16,16)
spatial_size = (32,32)
#hist_bins = 16
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True

t = time.time()
n_samples = 1000
random_idxs = np.random.randint(0,len(cars),n_samples)
#test_cars = np.array(cars)[random_idxs]
test_cars = cars
#test_notcars = np.array(notcars)[random_idxs]
test_notcars = notcars

car_features = extract_features(test_cars, 
	color_space=color_space, 
	spatial_size=spatial_size,
	hist_bins=hist_bins,
	orient=orient,
	pix_per_cell=pix_per_cell,
	cell_per_block=cell_per_block,
	hog_channel=hog_channel, spatial_feat=spatial_feat,
	hist_feat=hist_feat,hog_feat=hog_feat)

notcar_features = extract_features(test_notcars, 
	color_space=color_space, 
	spatial_size=spatial_size,
	hist_bins=hist_bins,
	orient=orient,
	pix_per_cell=pix_per_cell,
	cell_per_block=cell_per_block,
	hog_channel=hog_channel, spatial_feat=spatial_feat,
	hist_feat=hist_feat,hog_feat=hog_feat)

print(time.time()-t, "Seconds to compute features...")

X = np.vstack((car_features, notcar_features)).astype(np.float64)
X_scalar = StandardScaler().fit(X)

scaled_X = X_scalar.transform(X)

y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

rand_state = np.random.randint(0,100)
X_train, X_test, y_train, y_test = train_test_split(
	scaled_X, y, test_size=0.1, random_state=rand_state)


print("Using", orient, "orientations", pix_per_cell, "...")
print("Feature vector length", len(X_train[0]))

svc = LinearSVC()
t = time.time()
svc.fit(X_train, y_train)
print(time.time()-t, "Seconds to train SVM")
print("Test accuracy", svc.score(X_test, y_test))





