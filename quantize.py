import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
tflite_model_quant_file = tflite_models_dir/"/Users/arth/Desktop/tf-pose-estimation/models/graph/mobilenet_thin/graph.tflite"
tflite_model_quant_file.write_bytes(tflite_quant_model)
