import argparse
import logging
import time
import tensorflow as tf
import cv2

import cv2
import numpy as np

from tf_pose.estimator import PoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from tf_pose.tensblur.smoother import Smoother
from tf_pose import common

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

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

fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')

    args = parser.parse_args()

    tflite_model_file = get_graph_path(args.model)

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))

    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    while True:
        ret_val, image = cam.read()

        im = cv2.resize(image, (432,368), interpolation = cv2.INTER_AREA)
        im = im.reshape((1,368,432,3))
        im = im.astype(np.float32, copy=False)

        logger.debug('image process+')
        peaks, heatMat_up, pafMat_up = run_test(im, tflite_model_file)
        humans = PoseEstimator.estimate_paf(peaks[0], heatMat_up[0], pafMat_up[0])

        logger.debug('postprocess+')
        image = draw_humans(image, humans, imgcopy=False)

        logger.debug('show+')
        cv2.putText(image,"FPS: %f" % (1.0 / (time.time() - fps_time)),(10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) & 0xFF == ord('a'):
            break
        logger.debug('finished+')

    cv2.destroyAllWindows()
