############################### Info ###################################
"""
Author: Muhammad Umar Riaz
Date Updated: June 2017
"""
############################### Import #################################

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn import preprocessing, svm
from skimage import io, color, segmentation
from math import ceil
import os
import sys
import cv2
import glob
import time
import matplotlib.pyplot as plt
import numpy as np
import skimage
import caffe
import argparse

############################### Parser #################################

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input-dir', default='./Data/' ,help='Directory containing images (jpg format) to process')
parser.add_argument('-o','--output-dir', default='./Output/' ,help='Output directory')
parser.add_argument('--caffeNet-dir', default='./Tools/CaffeNet/', help='Directory containing caffeNet model')
parser.add_argument('--gpu-mode', default=False, help='Use gpu mode for caffe')
parser.add_argument('--training-features-dir', default='./Features/', help='Directory containing training features file')
parser.add_argument('--visual-output', default=True, help='Visual output of the processing (True for visualing, False otherwise)')
parser.add_argument('--visual-output-save', default=True, help='Save the visual output of the processing (True for saving, False otherwise)')
args = parser.parse_args()

############################ Functions #################################

def sliding_window(image, stepSize):
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            yield (x, y)

def load_image(filename, color=True):
    return skimage.img_as_float(skimage.io.imread(filename)).astype(np.float32)

def compute_uPattern():
    uPattern = []
    for n in range(256):
        nbin = bin(n)
        nstr = str(nbin)
        nstr = nstr[2:]
        byte = np.zeros(shape = [8])
        for x in range (8-len(nstr),8,1):
            byte[x] = nstr[x - 8 + len(nstr)]
        step = 0
        for bit in range(7):
            if byte[bit] != byte[bit+1]:
                step += 1
        if step < 3 :
            uPattern.append(n)
        uPattern_values = np.asarray(uPattern)
    return uPattern_values

def LTP_feature_extraction(image_block, reorder_vector, t, exp, uPattern_values):
    block_hist_upper = np.zeros(shape=[256])
    block_hist_lower = np.zeros(shape=[256])
    block_hist_uPattern_upper = np.zeros(shape=[59])
    block_hist_uPattern_lower = np.zeros(shape=[59])
    brows, bcols = image_block.shape
    # For each pixel in our image, ignoring the borders...
    for brow in range(1,brows-1):
        for bcol in range(1,bcols-1):
            # Get centre
            center = image_block[brow,bcol]
            # Get neighbourhood
            pixels = image_block[brow-1 : brow+2, bcol-1 : bcol+2]
            pixels = np.ravel(pixels)
            neighborhood = pixels[reorder_vector]
            # Get ranges and determine LTP
            low = center - t
            high = center + t
            block_LTP_out = np.zeros(neighborhood.shape)
            block_LTP_out[neighborhood < low] = -1
            block_LTP_out[neighborhood > high] = 1
            # Get upper and lower patterns -> LBP
            upper = np.copy(block_LTP_out)
            upper[upper == -1] = 0
            du = np.sum( pow(2, exp) * upper )
            lower = np.copy(block_LTP_out)
            lower[lower == 1] = 0
            lower[lower == -1] = 1
            dl = np.sum( pow(2, exp) * lower )
            if any(uPattern_values == du):
                block_hist_uPattern_upper[np.where(uPattern_values == du)[0][0]] += 1
            else:
                block_hist_uPattern_upper[58] += 1

            if any(uPattern_values == dl):
                block_hist_uPattern_lower[np.where(uPattern_values == dl)[0][0]] += 1
            else:
                block_hist_uPattern_lower[58] += 1
    fileRow = np.concatenate([block_hist_uPattern_lower, block_hist_uPattern_upper])
    return fileRow

###########################  Data Loading  #############################

org_images          = sorted(glob.glob(args.input_dir+"*.jpg"))
my_deploy_prototext = args.caffeNet_dir + 'deploy.prototxt'
my_caffemodel       = args.caffeNet_dir + 'model_caffenet.caffemodel'
my_meanfile         = args.caffeNet_dir + 'mean.npy'
feature_array       = np.load(args.training_features_dir + 'trainfeature.npy')
label_array         = np.load(args.training_features_dir + 'trainlabel.npy')
detect_step         = 33
seg_step            = 3
patch_dim           = 231
hair_thr            = 65
nonhair_thr         = 15

if args.gpu_mode:
    caffe.set_mode_gpu()
    caffe.set_device(0)
else:
    caffe.set_mode_cpu()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

####################### CaffeNet Initialization ########################

net = caffe.Net(my_deploy_prototext, my_caffemodel, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load(my_meanfile).mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)
net.blobs['data'].reshape(1,3,227,227)

################## Detection: Classifier Training ######################

print "--> Classifier training"

imp                 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 1)
feature_array       = imp.fit_transform(feature_array)
scalerDet           = preprocessing.StandardScaler().fit(feature_array)
feature_array       = scalerDet.transform(feature_array)
hair_idx            = np.where(label_array == 1.0)[0]
nonhair_idx         = np.where(label_array == 0.0)[0]
hair_samples_no     = hair_idx.shape[0]
nonhair_samples_no  = nonhair_idx.shape[0]

print "Hair samples: ", hair_samples_no, " NonHair samples: ", nonhair_samples_no

hair_array          = [feature_array[i] for i in hair_idx]
nonhair_array       = [feature_array[i] for i in nonhair_idx]
hair_label          = [1] * hair_samples_no
nonhair_label       = [0] * nonhair_samples_no
train_feature       = np.asarray(hair_array + nonhair_array)
train_label         = np.asarray(hair_label + nonhair_label)
my_clf_det          = RandomForestClassifier(n_estimators = 100)
my_clf_det          = my_clf_det.fit(train_feature, train_label)

################### Detection: Classifier Testing #######################

for n in range(len(org_images)):

    print "Processing: ", org_images[n].split('/')[-1:][0][:-4]

    org_image_rgb        = load_image(org_images[n])
    org_image            = color.rgb2gray(org_image_rgb)
    border_org_image_rgb = cv2.copyMakeBorder(org_image_rgb, patch_dim, patch_dim, patch_dim, patch_dim, cv2.BORDER_CONSTANT,value=cv2.mean(org_image_rgb)[:3])
    border_org_image     = cv2.copyMakeBorder(org_image, patch_dim, patch_dim, patch_dim, patch_dim, cv2.BORDER_CONSTANT,value=cv2.mean(org_image)[0])
    
    if args.visual_output_save or args.visual_output:
        image_footprint  = border_org_image_rgb.copy()
    output_image         = np.zeros(border_org_image_rgb.shape[:2], dtype = int)

    print "Hair Detection at patch-level"

    tic = time.clock()
    for (x, y) in sliding_window(border_org_image, stepSize = detect_step):
        if (y + patch_dim > border_org_image.shape[0]) or (x + patch_dim > border_org_image.shape[1]):
            continue
        image_block_rgb             = border_org_image_rgb[ y:y+patch_dim , x:x+patch_dim, :]
        image_block                 = border_org_image[ y:y+patch_dim , x:x+patch_dim]
        image_block_rgb_227         = image_block_rgb[2:229,2:229,:]
        net.blobs["data"].data[...] = transformer.preprocess("data", image_block_rgb_227)
        out                         = net.forward()
        fVector                     = net.blobs['fc7'].data[0].copy()
        feature_array               = imp.transform(fVector.reshape(1,-1))
        feature_array               = scalerDet.transform(feature_array)
        hair_prediction             = my_clf_det.predict(feature_array)

        if hair_prediction[0] == 1.0:
            output_image[ y:y+patch_dim, x:x+patch_dim ] += 1
            if args.visual_output_save or args.visual_output:
                image_footprint[y:y+patch_dim, x:x+patch_dim, :] -= 0.04
                hair_color = (0,0,255)
        elif hair_prediction[0] == 0.0:
            if args.visual_output_save or args.visual_output:
                hair_color = (0,255,0)

        if args.visual_output:
            clone = image_footprint.copy()
            cv2.rectangle(clone, (x, y ), (x+patch_dim, y+patch_dim), hair_color, 2)
            clone = clone[patch_dim:org_image.shape[0], patch_dim:org_image.shape[1]]
            cv2.imshow("Window", np.fliplr(clone.reshape(-1,3)).reshape(clone.shape))
            cv2.waitKey(1)
            time.sleep(0.025)
    toc = time.clock()

    print "Hair Detection completed"
    print "Processing time: ", toc-tic, "seconds"

    output_image           = output_image [patch_dim : patch_dim + org_image.shape[0], patch_dim : patch_dim + org_image.shape[1]]
    unique_val, counts_val = np.unique(output_image, return_counts=True)
    unique_counts_val      = dict(zip(unique_val, counts_val))

    if unique_counts_val.get(0) == output_image.shape[0] * output_image.shape[1] or output_image.max() <= 5:
        print "Processing finished. All pixels in the input image are labelled as nonhair."
        cv2.imwrite(args.output_dir + org_images[n].split('/')[-1][:-4] + "-" + "HairDetection-Hair-region.png", output_image.astype(np.int))
        continue
    elif unique_counts_val.get(49) == output_image.shape[0] * output_image.shape[1]:
        print "Processing finished. All pixels in the input image are labelled as hair."
        output_image[output_image == 49] = 255
        cv2.imwrite(args.output_dir + org_images[n].split('/')[-1][:-4] + "-" + "HairDetection-Hair-region.png", output_image.astype(np.int))
        continue

    hair_thr_relative    = int(ceil(float( output_image.max() * hair_thr ) / 100))
    nonhair_thr_relative = int(ceil(float( output_image.max() * nonhair_thr ) / 100))
    Hair_region          = output_image >= hair_thr_relative
    NonHair_region       = output_image <= nonhair_thr_relative

    cv2.imwrite(args.output_dir + org_images[n].split('/')[-1][:-4] + "-" + "HairDetection-Hair-region.png", Hair_region.astype(np.int)*255)
    cv2.imwrite(args.output_dir + org_images[n].split('/')[-1][:-4] + "-" + "HairDetection-NonHair-region.png", NonHair_region.astype(np.int)*255)

    if args.visual_output_save:
        image_footprint = image_footprint * 255
        cv2.imwrite(args.output_dir + org_images[n].split('/')[-1][:-4] + "-HairProb" + ".jpg", np.fliplr(image_footprint.reshape(-1,3)).reshape(image_footprint.shape))

    ############### Segmentation: Classifier Training ##################

    Block_size       = 25
    t                = 0.02
    reorder_vector   = np.array([0,1,2,5,8,7,6,3])
    exp              = np.array([7,6,5,4,3,2,1,0])
    uPattern_values  = compute_uPattern()
    Uncertain_region = (Hair_region + NonHair_region) != 1

    if np.count_nonzero(Uncertain_region) == 0:
        print "Processing finished. No uncertain region found."
        continue

    fList = []
    lList = []

    print "--> Overlapping 25x25 patch extraction from uncertain region"

    tic = time.clock()
    for (x, y) in sliding_window(Uncertain_region, stepSize=seg_step):
        if (y + Block_size > Uncertain_region.shape[0]) or (x + Block_size > Uncertain_region.shape[1]):
            continue
        patch_area_hair = np.count_nonzero(Hair_region[y:y+Block_size, x:x+Block_size])
        patch_area_nonhair = np.count_nonzero(NonHair_region[y:y+Block_size, x:x+Block_size])
        if patch_area_hair == (Block_size * Block_size):
            label = 1
        elif patch_area_nonhair == (Block_size * Block_size):
            label = 0
        else:
            label = "ImpureSample"
            continue

        image_block_rgb = org_image_rgb[y:y+Block_size, x:x+Block_size, :]
        image_block     = org_image[y:y+Block_size, x:x+Block_size]
        fVector         = LTP_feature_extraction(image_block, reorder_vector, t, exp, uPattern_values)
        fMeans          = cv2.mean(image_block_rgb)
        fVector         = np.append(fVector, fMeans[:3], 0)
        fList.append(fVector)
        lList.append(label)
    toc = time.clock()

    print "--> Processing time: ", toc-tic, "seconds"

    feature_array = np.asarray(fList)
    label_array   = np.asarray(lList)
    ones          = np.count_nonzero(label_array)
    zeros         = np.count_nonzero(1 - label_array)

    print "--> Classifier training"

    imp                = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 1)
    feature_array      = imp.fit_transform(feature_array)
    scalerSeg          = preprocessing.StandardScaler().fit(feature_array)
    feature_array      = scalerSeg.transform(feature_array)
    hair_idx           = np.where(label_array == 1.0)[0]
    nonhair_idx        = np.where(label_array == 0.0)[0]
    hair_samples_no    = hair_idx.shape[0]
    nonhair_samples_no = nonhair_idx.shape[0]

    print "--> Hair samples: ", hair_samples_no, " NonHair samples: ", nonhair_samples_no
    print "--> Balancing data"

    if hair_samples_no < nonhair_samples_no:
        nonhair_array = [feature_array[i] for i in np.random.choice(nonhair_idx, size=hair_samples_no, replace=False)]
        hair_array    = [feature_array[i] for i in np.random.choice(hair_idx, size=hair_samples_no, replace=False)]
        hair_label    = [1] * hair_samples_no
        nonhair_label = [0] * hair_samples_no
    elif nonhair_samples_no < hair_samples_no:
        hair_array    = [feature_array[i] for i in np.random.choice(hair_idx, size=nonhair_samples_no, replace=False)]
        nonhair_array = [feature_array[i] for i in np.random.choice(nonhair_idx, size=nonhair_samples_no, replace=False)]
        hair_label    = [1] * nonhair_samples_no
        nonhair_label = [0] * nonhair_samples_no

    print "--> Hair samples: ", len(hair_label), " NonHair samples: ", len(nonhair_label)

    train_feature = np.asarray(hair_array + nonhair_array)
    train_label   = np.asarray(hair_label + nonhair_label)
    my_clf        = svm.SVC()
    my_clf        = my_clf.fit(train_feature, train_label)

    ############### Segmentation: Classifier Testing ###################

    if args.visual_output:
        image_footprint= org_image_rgb.copy()

    Labels_img = 1 * Hair_region + 2 * NonHair_region

    print "Hair Segmentation at pixel-level"

    tic = time.clock()
    for (x, y) in sliding_window( Uncertain_region, stepSize=seg_step):
        if (y + Block_size > Uncertain_region.shape[0]) or (x + Block_size > Uncertain_region.shape[1]):
            continue

        Uncertain_patch_area = np.count_nonzero(Uncertain_region[ y+(Block_size-1)/2 : y+(Block_size-1)/2 + 3, x+(Block_size-1)/2 : x+(Block_size-1)/2 + 3])

        if Uncertain_patch_area == 0:
            continue

        image_block_rgb = org_image_rgb[y:y+Block_size, x:x+Block_size, :]
        image_block     = org_image[y:y+Block_size, x:x+Block_size]
        fVector         = LTP_feature_extraction(image_block, reorder_vector, t, exp, uPattern_values)
        fMeans          = cv2.mean(image_block_rgb)
        fVector         = np.append(fVector, fMeans[:3], 0)
        feature_array   = imp.transform(fVector.reshape(1,-1))
        feature_array   = scalerSeg.transform(feature_array)
        hair_prediction = my_clf.predict(feature_array)

        if hair_prediction[0] == 1.0:
            Labels_img[ y+(Block_size-1)/2 : y+(Block_size-1)/2 + 3, x+(Block_size-1)/2 : x+(Block_size-1)/2 + 3 ] = 3
            if args.visual_output:
                image_footprint[y+(Block_size-1)/2 : y+(Block_size-1)/2 + 3, x+(Block_size-1)/2 : x+(Block_size-1)/2 + 3, :] -= 0.04
                hair_color = (0,0,255)

        elif hair_prediction[0] == 0.0:
            Labels_img[ y+(Block_size-1)/2 : y+(Block_size-1)/2 + 3, x+(Block_size-1)/2 : x+(Block_size-1)/2 + 3 ] = 4
            if args.visual_output:
                image_footprint[y+(Block_size-1)/2 : y+(Block_size-1)/2 + 3, x+(Block_size-1)/2 : x+(Block_size-1)/2 + 3, :] += 0.04
                hair_color = (0,255,0)

        if args.visual_output:
            clone = image_footprint.copy()
            cv2.rectangle(clone, ( x+(Block_size-1)/2, y+(Block_size-1)/2 ), (x+(Block_size-1)/2 + 3, y+(Block_size-1)/2 + 3), hair_color, 2)
            cv2.imshow("Window", np.fliplr(clone.reshape(-1,3)).reshape(clone.shape))
            cv2.waitKey(1)
            time.sleep(0.025)

    toc = time.clock()

    print "Hair Segmentstion completed"
    print "Processing time :", toc - tic, "seconds"
    print "Post-Processing"

    tic = time.clock()
    Labels_img[Labels_img == 0] = 4
    Labels_img  = Labels_img - 1
    nseg = int((org_image_rgb.shape[0] + org_image_rgb.shape[1] ) * 0.8)
    
    labels_post                  = segmentation.slic(org_image_rgb, n_segments=nseg, sigma=5, enforce_connectivity=True, slic_zero=True)
    labels_post                  = labels_post + 4
    labels_post[Labels_img == 0] = 0
    labels_post[Labels_img == 1] = 1
    labels_post_org              = labels_post.copy()

    for (i, segVal) in enumerate(np.unique(labels_post_org)):
        if segVal == 0 or segVal == 1:
            continue

        mask_seg                            = np.zeros(org_image_rgb.shape[:2], dtype = int)
        mask_seg[labels_post_org == segVal] = 1
        mask_newhair                        = np.zeros(org_image_rgb.shape[:2], dtype = int)
        mask_newhair[Labels_img == 2]       = 1
        mask_newnonhair                     = np.zeros(org_image_rgb.shape[:2], dtype = int)
        mask_newnonhair[Labels_img == 3]    = 1
        Hair_overlap                        = np.logical_and(mask_seg, mask_newhair)
        NonHair_overlap                     = np.logical_and(mask_seg, mask_newnonhair)

        if np.count_nonzero(Hair_overlap) < np.count_nonzero(NonHair_overlap):
            labels_post[labels_post == segVal] = 3
        else:
            labels_post[labels_post == segVal] = 2

        if args.visual_output:
            Seg_image = color.label2rgb(labels_post, org_image_rgb, kind='overlay')
            Seg_image_boundaries = segmentation.mark_boundaries(Seg_image, labels_post, (0, 0, 0))
            cv2.imshow("hairs", np.fliplr(Seg_image_boundaries.reshape(-1,3)).reshape(Seg_image_boundaries.shape))
            cv2.waitKey(1)
            time.sleep(0.025)
    toc = time.clock()

    print "Post-Processing completed"
    print "Processing time :", toc - tic, "seconds"

    Seg_image = color.label2rgb(labels_post, org_image_rgb, kind='overlay')
    Seg_image_boundaries = segmentation.mark_boundaries(Seg_image, labels_post, (0, 0, 0))
    Labels_img_bin = labels_post.copy() + 1
    Labels_img_bin[Labels_img_bin == 3] = 1
    Labels_img_bin[Labels_img_bin == 4] = 2
    Labels_img_bin[Labels_img_bin == 2] = 0

    np.save(args.output_dir + org_images[n].split('/')[-1][:-4] + "-" + "HairSegmentation-Result.npy", Labels_img_bin)
    cv2.imwrite(args.output_dir + org_images[n].split('/')[-1][:-4] + "-" + "HairSegmentation-Hair-region.png", Labels_img_bin.astype(np.int)*255)
    Seg_image_boundaries = Seg_image_boundaries * 255
    cv2.imwrite(args.output_dir + org_images[n].split('/')[-1][:-4] + "-" + "HairSegmentation-Result.png", np.fliplr(Seg_image_boundaries.reshape(-1,3)).reshape(Seg_image_boundaries.shape))

    print "Processing finished."
