from keras import backend as K
from keras.engine.topology import Layer
from theano.tensor.shared_randomstreams import RandomStreams

import theano
import numpy as np	

def crop_img(input_img, location, size):
	return  input_img[location[0]:location[0]+size[0], location[1]:location[1]+size[1]]

class RandomImageSliceLayer(Layer):
	def __init__(self, output_img_size, **kwargs):

		print('RandomImageSliceLayer()) output_shape: %s' % str(output_img_size))
		self.output_img_size = output_img_size
		super(RandomImageSliceLayer, self).__init__(**kwargs)
		self.rng = RandomStreams()

	def build(self, input_shape):
		print('RandomImageSliceLayer.build() input_shape: %s' % str(input_shape))
		self.input_img_size = (input_shape[2], input_shape[3])
		self.batch_size = self.input_img_size[0]
		self.x_offset_range = self.input_img_size[0] - self.output_img_size[0]+1
		self.y_offset_range = self.input_img_size[1] - self.output_img_size[1]+1
		print('offset ranges: %d, %d' % (self.x_offset_range, self.y_offset_range))

	def call(self, x, mask=None):	
		if 0:
			x_offsets = self.rng.random_integers((x.shape[0],2), low=0, high=self.x_offset_range)
			y_offsets_subtensor = x_offsets[:,1]
			offsets = theano.tensor.set_subtensor(y_offsets_subtensor, self.rng.random_integers((x.shape[0],), low=0, high=self.y_offset_range), )
		elif 0:
			offsets_np = np.array([[x, y] for x in range(self.x_offset_range) for y in range(self.y_offset_range)])
			#print(offsets_np)
			offsets = theano.shared(offsets_np)
		else:
			crop_regions = []
			for y_start in [0, 4, 8]:
				for x_start in [0, 8, 16, 24]:
					crop_regions.append([x_start, y_start])

			repeat_factor = int(np.ceil(float(self.batch_size) / len(crop_regions)))
			#print('repeat factor: %d' % repeat_factor)
			offsets_np = np.tile(np.array(crop_regions), (repeat_factor, 1))[:self.batch_size]
			#print(offsets_np)
			offsets = theano.shared(offsets_np)

		x_squeezed = K.squeeze(x, 1)

		sizes = np.array([self.output_img_size[0], self.output_img_size[1]])
		#print(sizes)
		sizes = theano.shared(sizes)

		r1, u1 = theano.map(crop_img,
			sequences = [x_squeezed, offsets],
			non_sequences = [sizes],
			name='crop image map func'
			)

		result = r1.reshape((x.shape[0], x.shape[1], self.output_img_size[0], self.output_img_size[1]))
		return result

	def get_output_shape_for(self, input_shape):  
		#result = (input_shape[0], self.batch_size, self.output_img_size[0], self.output_img_size[1])
		result = (input_shape[0], input_shape[1], self.output_img_size[0], self.output_img_size[1])
		return result

if __name__ == "__main__":
	from keras.models import Sequential

	print('main()')

	cropped_size = (40, 40)
	raw_size = (1, 64, 48)
	
	n_samples = (raw_size[1] - cropped_size[0] + 1) * (raw_size[2] - cropped_size[1] + 1)
	print('num samples: %d' % n_samples)
	
	model = Sequential()
	model.add(RandomImageSliceLayer(input_shape = raw_size, output_img_size = cropped_size))


	model.summary()
	model.compile(optimizer='sgd', loss='categorical_crossentropy')

	single_test_data = np.arange(raw_size[2]*raw_size[1]*raw_size[0]).reshape(raw_size).astype('float32')	
	test_data = np.tile(single_test_data, (n_samples, 1, 1))
	print(test_data.shape)
	print((1, n_samples, raw_size[1], raw_size[2]))

	test_data = test_data.reshape((n_samples, 1, raw_size[1], raw_size[2]))

	print(test_data[0][0])
	print(test_data.shape)

	cropped_data = model.predict(test_data, batch_size=n_samples)
	print(cropped_data)
	print(cropped_data.shape)