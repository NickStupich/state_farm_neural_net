from keras import backend as K
from keras.engine.topology import Layer
from theano.tensor.shared_randomstreams import RandomStreams

import theano

def crop_img(input_img, size, location):
	print('crop_img')
	#print(a_location.shape)
	print(location)
	print(input_img.shape)
	#result =  input_img[location_x[0]:location_x[0]+size_x, location_y[0]:location_y[0]+size_y]
	result =  input_img[location[0]:location[0]+size[0], location[1]:location[1]+size[1]]
		#result =  input_img[2:2+size_y, 2:2+size_x]

	return result



class RandomImageSliceLayer(Layer):
	def __init__(self, output_img_size, **kwargs):
		self.output_img_size = output_img_size
		super(RandomImageSliceLayer, self).__init__(**kwargs)
		self.rng = RandomStreams()

	def build(self, input_shape):
		print('RandomImageSliceLayer.build() input_shape: %s' % str(input_shape))
		self.input_img_size = (input_shape[1], input_shape[2])
		self.x_offset_range = self.input_img_size[0] - self.output_img_size[0]
		self.y_offset_range = self.input_img_size[1] - self.output_img_size[1]

	def call(self, x, mask=None):
	
		# x_offsets = self.rng.random_integers((x.shape[0],2), low=0, high=self.x_offset_range)
		# y_offsets_subtensor = x_offsets[:,1]
		# offsets = theano.tensor.set_subtensor(y_offsets_subtensor, self.rng.random_integers((x.shape[0],), low=0, high=self.y_offset_range), )

		# offsets = self.rng.random_integers((x.shape[0],2), low=0, high=self.x_offset_range)
		# x_offsets_subtensor = offsets[:,0]
		# offsets = theano.tensor.set_subtensor(x_offsets_subtensor, np.tile(np.arange(self.x_offset_range+1), self.y_offset_range+1))
		# y_offsets_subtensor = offsets[:,1]
		# offsets = theano.tensor.set_subtensor(y_offsets_subtensor, np.repeat(np.arange(self.y_offset_range+1), self.x_offset_range+1))

		#offsets_np = np.array([[y, x] for x in range(self.x_offset_range+1) for y in range(self.y_offset_range+1)])
		offsets_np = np.array([[2, 2, 0], [1, 1, 0], [0, 0, 0]])#.reshape(9, 2)
		print(offsets_np)
		# print(offsets_np)
		# print(offsets_np.shape)
		offsets = theano.shared(offsets_np)

		#offsets_x = theano.shared(np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]))
		#offsets_y = theano.shared(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]))

		#print(offsets.eval())

		#x_squeezed = K.squeeze(x, 1)
		#x_squeezed = x.reshape((x.shape[0]*x.shape[1], self.input_img_size[0], self.input_img_size[1]))
		# print(x_squeezed.eval())

		# r1, u1 = theano.map(fn = lambda input_img, a_location, size_x, size_y: input_img[a_location[1]:a_location[1]+size_y, a_location[0]:a_location[0]+size_x],
		# 	sequences = [x_squeezed, offsets],
		# 	non_sequences = [self.output_img_size[0], self.output_img_size[1]],
		# 	)

		sizes = np.tile(np.array([self.output_img_size[0], self.output_img_size[1], 17]), 9)#.reshape(9, 2)
		print(sizes)
		sizes = theano.shared(sizes)

		r1, u1 = theano.scan(crop_img,
			sequences = [x, sizes, offsets],
			non_sequences = [],
			name='crop image map func'
			)

		#result = r1.reshape((x.shape[0], x.shape[1], self.output_img_size[0], self.output_img_size[1]))
		#return result

		return r1

	def get_output_shape_for(self, input_shape):  
		result = (input_shape[0], self.output_img_size[0], self.output_img_size[1])
		return result

if __name__ == "__main__":
	from keras.models import Sequential
	import numpy as np

	print('main()')

	cropped_size = (3, 3)
	raw_size = (5, 5)
	
	n_samples = (raw_size[0] - cropped_size[0] + 1) * (raw_size[1] - cropped_size[1] + 1)
	
	model = Sequential()
	model.add(RandomImageSliceLayer(input_shape = raw_size, output_img_size = cropped_size))


	model.summary()
	model.compile(optimizer='sgd', loss='categorical_crossentropy')

	single_test_data = np.arange(raw_size[0]*raw_size[1]).reshape(raw_size).astype('float32')	
	test_data = np.tile(single_test_data, (n_samples, 1)).reshape((n_samples, raw_size[0], raw_size[1]))

	print(test_data[1][0])
	print(test_data.shape)

	cropped_data = model.predict(test_data, batch_size=2)
	print(cropped_data)
	print(cropped_data.shape)