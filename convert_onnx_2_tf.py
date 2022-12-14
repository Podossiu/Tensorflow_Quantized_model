import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("Test.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("saved_model")
