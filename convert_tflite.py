tflite_convert \
--graph_def_file='/Users/arth/Desktop/tf-pose-estimation/models/graph/mobilenet_thin/graph_opt.pb' \
--output_file='/Users/arth/Desktop/tf-pose-estimation_tflite/models/graph/mobilenet_thin/graph.tflite' \
--input_format=TENSORFLOW_GRAPHDEF \
--output_format=TFLITE \
--input_shape=1,368,432,3 \
--input_array=image \
--output_array=Openpose/concat_stage7 \
--inference_type=FLOAT \
--input_data_type=FLOAT


export dir1="/Users/arth/Desktop/tf-pose-estimation_tflite/models/graph/mobilenet_thin"
tflite_convert
--output_file=$dir1/graph.tflite
--graph_def_file=$dir1/graph_opt.pb
--input_array=image
--input_shape=1,368,432,3
--output_array=Openpose/concat_stage7
