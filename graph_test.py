import graph
import ops
import numpy as np


graph = graph.TensorGraph()

in1 = graph.add_source_node('input1')
in1 = graph.add_op(ops.stop_gradient(), in1)
scale = graph.add_source_node('scale')
bias = graph.add_source_node('bias')

target = graph.add_source_node('target')
target2 = graph.add_op(ops.stop_gradient(), target)

in2 = graph.add_op(ops.multiply(), [in1, scale])
out1 = graph.add_op(ops.sum(), [in2, bias])
pred = graph.add_terminal_node(out1, 'out1')

error_raw = graph.add_op(ops.subtract(), [out1, target2])
error = graph.add_op(ops.square(), error_raw)
total_error = graph.add_op(ops.reduce_sum(), error)
total_error = graph.add_terminal_node(total_error, 'total_error')


init_scale = 0.1
init_bias = -0.1

true_scale = 0.2
true_bias = -0.5

var_scale = init_scale
var_bias = init_bias

print(true_scale)
print(true_bias)

print(var_bias)
print(var_scale)

for i in range(5000):
    x = np.random.rand(10)

    y = x * true_scale + true_bias
    y += np.random.normal(0, 0.01, 10)

    total_error.eval(feed_dict={'input1': x, 'scale': var_scale, 'bias': var_bias, 'target': y})

    pred.eval(feed_dict={'input1': x, 'scale': var_scale, 'bias': var_bias, 'target': y})
    scale_g, bias_g = total_error.back(to=[scale, bias])
    scale_g = np.mean(scale_g)
    bias_g = np.mean(bias_g)
    var_scale -= 0.01 * scale_g
    var_bias -= 0.01 * bias_g

    graph.reset()

print(var_scale)
print(var_bias)