from keras import backend as K
from keras.engine.topology import Layer
from theano.tensor.shared_randomstreams import RandomStreams

import theano

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
		if 0:
			x_offset = self.rng.random_integers((1,), low=0, high=self.x_offset_range-1)[0]
			y_offset = self.rng.random_integers((1,), low=0, high=self.y_offset_range-1)[0]
			
			#x_offset = 12
			#y_offset = 4
			
			result = x[:, :, x_offset:x_offset + self.output_img_size[0], y_offset:y_offset + self.output_img_size[1]]
			return result

		else:
			#offsets = self.rng.random_integers((x.shape[0], 2), low=0, high=8)
			
			x_offsets = self.rng.random_integers((x.shape[0],2), low=0, high=self.x_offset_range-1)
			y_offsets_subtensor = x_offsets[:,1]
			offsets = theano.tensor.set_subtensor(y_offsets_subtensor, self.rng.random_integers((x.shape[0],), low=0, high=self.y_offset_range-1), )


			x_squeezed = K.squeeze(x, 1)

			r1, u1 = theano.map(fn = lambda input_img, a_location, size_x, size_y: input_img[a_location[0]:a_location[0]+size_x, a_location[1]:a_location[1]+size_y],
				sequences = [x_squeezed, offsets],
				non_sequences = [self.output_img_size[0], self.output_img_size[1]],
				)

			result = r1.reshape((x.shape[0], x.shape[1], self.output_img_size[0], self.output_img_size[1]))

			return result

	def get_output_shape_for(self, input_shape):  
		result = (input_shape[0], input_shape[1], self.output_img_size[0], self.output_img_size[1])
		return result



if __name__ == "__main__":
	print('main()')
	#result = doRandomCrop(in)
