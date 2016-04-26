from keras import backend as K
from keras.engine.topology import Layer
# from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams

class RandomImageSliceLayer(Layer):
    def __init__(self, output_img_size, **kwargs):
        self.output_img_size = output_img_size
        super(RandomImageSliceLayer, self).__init__(**kwargs)
        self.rng = RandomStreams()

    def build(self, input_shape):
        print('RandomImageSliceLayer.build() input_shape: %s' % str(input_shape))
        self.input_img_size = (input_shape[2], input_shape[3])
        self.x_offset_range = self.input_img_size[0] - self.output_img_size[0]
        self.y_offset_range = self.input_img_size[1] - self.output_img_size[1]

    def call(self, x, mask=None):
        x_offset = self.rng.random_integers((1,), low=0, high=self.x_offset_range-1)[0]
        y_offset = self.rng.random_integers((1,), low=0, high=self.y_offset_range-1)[0]
        result = x[:, :, x_offset:x_offset + self.output_img_size[0], y_offset:y_offset + self.output_img_size[1]]
        
        #x_offsets = self.rng.random_integers((,), low=0, high=self.x_offset_range-1)[0]
        #y_offsets = self.rng.random_integers((1,), low=0, high=self.y_offset_range-1)[0]

        return result

    def get_output_shape_for(self, input_shape):  
        result = (input_shape[0], input_shape[1], self.output_img_size[0], self.output_img_size[1])
        return result

