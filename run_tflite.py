'''
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tutorials/post_training_quant.ipynb
'''

import sys
import os
import argparse

import cv2
from tf_pose import common

import tensorflow as tf

import numpy as np

from tf_pose.tensblur.smoother import Smoother
from tf_pose.estimator import PoseEstimator
from tf_pose.networks import get_graph_path, model_wh


def run_test(image, model):
    persistent_sess = tf.Session()
    tensor_image = image
    interpreter = tf.contrib.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    interpreter.set_tensor(input_index, tensor_image)
    interpreter.invoke()
    tensor_output = interpreter.get_tensor(output_index)

    upsample_size = [116, 108]
    tensor_heatMat_up = tf.image.resize_area(tensor_output[:, :, :, :19], upsample_size,align_corners=False, name='upsample_heatmat')
    tensor_pafMat_up = tf.image.resize_area(tensor_output[:, :, :, 19:], upsample_size,align_corners=False, name='upsample_pafmat')
    smoother = Smoother({'data': tensor_heatMat_up}, 25, 3.0)
    gaussian_heatMat = smoother.get_output()

    max_pooled_in_tensor = tf.nn.pool(gaussian_heatMat, window_shape=(3, 3), pooling_type='MAX', padding='SAME')
    tensor_peaks = tf.where(tf.equal(gaussian_heatMat, max_pooled_in_tensor), gaussian_heatMat,
                                 tf.zeros_like(gaussian_heatMat))

    persistent_sess.close()
    persistent_sess = tf.Session()
    persistent_sess.run(tf.global_variables_initializer())
    peaks, heatMat_up, pafMat_up = persistent_sess.run(  [tensor_peaks, tensor_heatMat_up, tensor_pafMat_up])
    return peaks, heatMat_up, pafMat_up

def draw_humans(npimg, humans, imgcopy=False):
    if imgcopy:
        npimg = np.copy(npimg)
    image_h, image_w = npimg.shape[:2]
    centers = {}
    for human in humans:
        # draw point
        for i in range(common.CocoPart.Background.value):
            if i not in human.body_parts.keys():
                continue

            body_part = human.body_parts[i]
            center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
            centers[i] = center
            cv2.circle(npimg, center, 3, common.CocoColors[i], thickness=3, lineType=8, shift=0)

        # draw line
        for pair_order, pair in enumerate(common.CocoPairsRender):
            if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                continue

            # npimg = cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)
            cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)

    return npimg

def plot_human(humans, image):

    import matplotlib.pyplot as plt

    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    a.set_title('Result')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__== "__main__":

    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image', type=str, default='./images/p1.jpg')
    parser.add_argument('--model', type=str, default='cmu',help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
        
    args = parser.parse_args()
    
    tflite_model_file = get_graph_path(args.model)
    image_path = args.image
    
    import cv2
    im = cv2.imread(image_path)
    im = cv2.resize(im, (432,368), interpolation = cv2.INTER_AREA)
    im = im.reshape((1,368,432,3))
    im = im.astype(np.float32, copy=False)

    # # models
    image1 = common.read_imgfile(image_path, None, None)
    peaks, heatMat_up, pafMat_up = run_test(im, tflite_model_file)
    humans = PoseEstimator.estimate_paf(peaks[0], heatMat_up[0], pafMat_up[0])
    image = draw_humans(image1, humans, imgcopy=False)
    plot_human(humans, image)

    print("end1")
