import numpy as np
_sum = sum

class TensorOp(object):
    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def grad(self, acts, prev_grads):
        raise NotImplementedError()

class sum(TensorOp):
    def __init__(self):
        super().__init__(name='sum_op')

    def __call__(self, *inputs):
        return _sum(inputs)

    def grad(self, inputs, prev_grads):
        grad_wrt_acts = []
        partial_wrt_inputs = [1.0 for x in inputs]
        grad_wrt_inputs = [p*prev_grads for p in partial_wrt_inputs]
        return grad_wrt_acts, grad_wrt_inputs

class reduce_sum(TensorOp):
    def __init__(self, axis=None):
        self.axis=axis
        super().__init__(name='sum_op')

    def __call__(self, x):
        return np.sum(x, axis=self.axis)

    def grad(self, x, prev_grads):
        grad_wrt_acts = []
        partial_wrt_inputs = [np.ones_like(x)]
        grad_wrt_inputs = [p*prev_grads for p in partial_wrt_inputs]
        return grad_wrt_acts, grad_wrt_inputs

class subtract(TensorOp):
    def __init__(self):
        super().__init__(name='subtract_op')

    def __call__(self, *inputs):
        return inputs[0] - inputs[1]

    def grad(self, inputs, prev_grads):
        grad_wrt_acts = []
        partial_wrt_inputs = [1.0, -1.0]
        grad_wrt_inputs = [p*prev_grads for p in partial_wrt_inputs]
        return grad_wrt_acts, grad_wrt_inputs

class square(TensorOp):
    def __init__(self):
        super().__init__(name='square_op')

    def __call__(self, x):
        return x ** 2

    def grad(self, x, prev_grads):
        grad_wrt_acts = []
        partial_wrt_inputs = [2*x]
        grad_wrt_inputs = [p*prev_grads for p in partial_wrt_inputs]
        return grad_wrt_acts, grad_wrt_inputs

class multiply(TensorOp):
    def __init__(self, name='multiply_by_c'):
        super().__init__(name=name)

    def __call__(self, *inputs):
        return inputs[0]*inputs[1]

    def grad(self, inputs, prev_grads):
        grad_wrt_acts = inputs[0]*prev_grads
        partial_wrt_inputs = [inputs[1], inputs[0]]
        grad_wrt_inputs = [p*prev_grads for p in partial_wrt_inputs]
        return grad_wrt_acts, grad_wrt_inputs

class multiply_by_constant(TensorOp):
    def __init__(self, c, name='multiply_by_c'):
        self.c = c
        super().__init__(name=name)

    def __call__(self, x):
        return self.c*x

    def grad(self, inputs, prev_grads):
        grad_wrt_acts = inputs[0]*prev_grads
        partial_wrt_inputs = self.c
        grad_wrt_inputs = [prev_grads * partial_wrt_inputs,]
        return grad_wrt_acts, grad_wrt_inputs

class identity(TensorOp):
    def __init__(self, name='identity'):
        super().__init__(name=name)

    def __call__(self, x):
        return x

    def grad(self, inputs, prev_grads):
        grad_wrt_acts = []
        partial_wrt_inputs = 1.0
        grad_wrt_inputs = [prev_grads * partial_wrt_inputs,]
        return grad_wrt_acts, grad_wrt_inputs


class stop_gradient(TensorOp):
    def __init__(self, name='stop_gradient'):
        super().__init__(name=name)

    def __call__(self, x):
        return x

    def grad(self, inputs, prev_grads):
        grad_wrt_acts = []
        partial_wrt_inputs = 0.0
        grad_wrt_inputs = [prev_grads * partial_wrt_inputs,]
        return grad_wrt_acts, grad_wrt_inputs