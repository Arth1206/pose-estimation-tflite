import tensorflow as tf

path = '/Users/arth/Desktop/tf-pose-estimation_tflite/models/graph/mobilenet_thin/graph_opt.pb'

#input_shapes = {'image':[1,368,432,3]}

#converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(path, ['image'], ['Openpose/concat_stage7'], input_shapes)
#converter.optimizations = [tf.contrib.lite.Optimize.OPTIMIZE_FOR_SIZE]
#converter.post_training_quantize = True
#converter.inference_type = tf.contrib.lite.constants.QUANTIZED_UINT8
#converter.quantized_input_stats = {'image':(127,1.0/128)}
#converter.allow_custom_ops = True
#converter.default_ranges_stats = (0,6)


input_shapes = {'image':[1,368,432,3]}
input_arrays = ["image"]
output_arrays = ["Openpose/concat_stage7"]

converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(path, input_arrays, output_arrays, input_shapes)

tflite_model = converter.convert()
open("/Users/arth/Desktop/tf-pose-estimation_tflite/models/graph/mobilenet_thin/graph1.tflite","wb").write(tflite_model)

