import os
import time
import glob
import cv2
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label

from external import extract_features
from external import convert_color
from external import get_hog_features
from external import bin_spatial
from external import color_hist
from external import apply_threshold
from external import draw_labeled_bboxes
from train import train

############
# Hyper #
############
color_space = "YCrCb" #"RGB"
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" #0
spatial_size = (32,32) #(16,16)
hist_bins = 32 #16
spatial_feat = True
hist_feat = True
hog_feat = True
############
# Globals:
X_scalar = None
svc = None
trained = False



def generate_image(img):
    out_img, heat_map = find_cars(img)
    heat_map = apply_threshold(heat_map, 0)
    labels = label(heat_map)
    draw_image = draw_labeled_bboxes(np.copy(img), labels)
    return draw_image

def find_cars(img, scale=1.5, ystart=400, ystop=656, pix_per_cell=8, orient=9, cell_per_block=2, spatial_size=(32,32), hist_bins=32):
    if trained == False:
        global X_scalar
        global svc
        global trained
        X_scalar, svc = train()
        trained = True

    draw_img = np.copy(img)
    heatmap = np.zeros_like(img[:,:,0])
    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv="RGB2YCrCb")
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    nxblocks = (ch1.shape[1]//pix_per_cell) - 1
    nyblocks = (ch1.shape[0]//pix_per_cell)-1
    nfeat_per_block = orient*cell_per_block**2
    window = 64
    nblocks_per_window = (window//pix_per_cell)-1
    cells_per_step = 2
    nxsteps = (nxblocks - nblocks_per_window) //cells_per_step
    nysteps = (nyblocks - nblocks_per_window) //cells_per_step

    hog1 = get_hog_features(ch1, orient, pix_per_cell,
                           cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell,
                           cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell,
                           cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            test_features = X_scalar.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1,-1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0))   
                #img_boxes.append(((xbox_left,ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart, xbox_left:xbox_left+win_draw]+=1

    return draw_img, heatmap
