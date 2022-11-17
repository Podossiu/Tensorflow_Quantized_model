import tensorflow as tf
import pathlib
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()


tflite_model_dir = pathlib.Path("./tflite_models")
tflite_model_dir.mkdir(exist_ok = True, parents = True)

tflite_model_quant_file = tflite_model_dir/"tflite_model.tflite"
tflite_model_quant_file.write_bytes(tflite_quant_model)

