__author__ = 'Ldaze'

from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.keras import backend as bd


class AttentionOnSteps(Layer):

    def __init__(self, **kwargs):
        self.output_dim = None
        self.kernel = None
        super(AttentionOnSteps, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[2]
        self.kernel = self.add_weight(
            name='kernel',
            shape=(3, input_shape[2], self.output_dim),
            initializer='uniform',
            trainable=True
        )

        super(AttentionOnSteps, self).build(input_shape)

    def call(self, x, **kwargs):
        wq = bd.dot(x, self.kernel[0])
        wk = bd.dot(x, self.kernel[1])
        wv = bd.dot(x, self.kernel[2])
        qk = bd.batch_dot(wq, bd.permute_dimensions(wk, [0, 2, 1]))
        qk = qk / (self.output_dim**0.5)
        qk = bd.softmax(qk)
        new_vector = bd.batch_dot(qk, wv)
        return new_vector

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.output_dim


class AttentionOnDims(Layer):
    def __init__(self, **kwargs):
        self.dim = None
        self.kernel = None
        super(AttentionOnDims, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dim = input_shape[2]
        self.kernel = self.add_weight(
            name='kernel',
            shape=(1, 1, input_shape[1]),
            initializer='uniform',
            trainable=True
        )
        super(AttentionOnDims, self).build(input_shape)

    def call(self, x, **kwargs):
        w = bd.dot(bd.permute_dimensions(x, [0, 2, 1]), bd.permute_dimensions(self.kernel[0], [1, 0]))
        w = bd.permute_dimensions(w, [0, 2, 1])
        w = bd.softmax(w)
        return x*w

    def compute_output_shape(self, input_shape):
        return input_shape
