import theano
import theano.tensor as T
import numpy as np

if 0:
	location = T.imatrix("location")
	x = T.ftensor4("x")
	k_x = T.iscalar('k_x')
	k_y = T.iscalar('k_y')

	def set_value_at_position(a_location, input_values, k_x, k_y):
		subtensor = input_values[:, a_location[0]:a_location[0] + k_x, a_location[1]:a_location[1] + k_y]
		return subtensor
		
	result, updates = theano.scan(fn=set_value_at_position,
									outputs_info=None,
									sequences=[location, x],
									non_sequences=[k_x, k_y]
									)

	assign_values_at_positions = theano.function(inputs=[location, x, k_x, k_y], outputs=result)

	dim_x = 7
	dim_y = 5
	n = 3
	k_x_test = 5
	k_y_test = 3
	location_test = np.asarray([[1, 1], [0, 0], [1, 2], [0, 0], [0, 0]], dtype=np.int32)
	x_test = np.reshape(np.arange(dim_x * dim_y * n, dtype=np.float32), (1, n,dim_x, dim_y))

	print(x_test)
	print(x_test.shape)
	result = assign_values_at_positions(location_test, x_test, k_x_test, k_y_test)
	print(result)
	print(result.shape)

if 0:
	location = T.imatrix("location")
	img_batch = T.ftensor3("img_batch")
	x = T.ftensor4("x")
	size_x = T.iscalar('size_x')
	size_y = T.iscalar('size_y')

	def get_sub_image(a_location, input_img, size_x, size_y):
		result = input_img[a_location[0]:a_location[0]+size_x, a_location[1]:a_location[1]+size_y]
		return result

	r1, u1 = theano.map(fn = get_sub_image,
						sequences = [location, img_batch],
						non_sequences = [size_x, size_y],
						)

	crop_image_batch = theano.function(inputs=[img_batch, location, size_x, size_y], outputs = r1, updates=u1)

	def crop_image_batch_python(img_batch, location, size_x, size_y):
		return crop_image_batch(img_batch, location, size_x, size_y)

	r2, u2 = theano.map(fn = crop_image_batch_python,
						sequences = [x],
						non_sequences = [location, size_x, size_y]
						)


	# crop_images = theano.function(inputs=[location, x, size_x, size_y], outputs = r2, updates=u2)

	dim_x = 7
	dim_y = 5
	n = 3
	size_x_test = 5
	size_y_test = 3
	location_test = np.asarray([[1, 1], [0, 0], [1, 2]], dtype=np.int32)

	img_batch_test = np.reshape(np.arange(dim_x * dim_y * n, dtype=np.float32), (n,dim_x, dim_y))
	print(img_batch_test)
	print(location_test)
	batch_result = crop_image_batch(img_batch_test, location_test, size_x_test, size_y_test)

	print(batch_result)

	x_test = np.array([img_batch_test, img_batch_test*2])

	print(x_test)
	print(x_test.shape)
	result = crop_images(location_test, x_test, size_x_test, size_y_test)
	print(result)
	print(result.shape)

if 1:
	location = T.imatrix("location")
	img_batch = T.ftensor3("img_batch")
	x = T.ftensor4("x")
	size_x = T.iscalar('size_x')
	size_y = T.iscalar('size_y')

	def get_sub_image(input_img, a_location, size_x, size_y):
		# result = T.zeros((input_imgs.shape[0], size_x, size_y))
		# for i in range(input_imgs.shape[0]):
		# 	T.set_subtensor(result_subtensor, input_imgs[i, a_location[0]:a_location[0]+size_x, a_location[1]:a_location[1]+size_y])

		result = input_img[a_location[0]:a_location[0]+size_x, a_location[1]:a_location[1]+size_y]
		return result

	r1, u1 = theano.map(fn = get_sub_image,
						sequences = [img_batch, location],
						non_sequences = [size_x, size_y],
						)

	crop_image_batch = theano.function(inputs=[img_batch, location, size_x, size_y], outputs = r1, updates=u1)

	dim_x = 7
	dim_y = 5
	n = 3
	size_x_test = 5
	size_y_test = 3
	location_test = np.asarray([[1, 1], [0, 0], [1, 2]], dtype=np.int32)

	img_batch_test = np.reshape(np.arange(dim_x * dim_y * n, dtype=np.float32), (n, 1, dim_x, dim_y))
	x_test = np.array([img_batch_test, img_batch_test*2])
	print(img_batch_test)
	print(location_test)

	batch_result = crop_image_batch(img_batch_test[:, 0], location_test, size_x_test, size_y_test)
	batch_result = batch_result.reshape(n, 1, size_x_test, size_y_test)
	print(batch_result)
	print(batch_result.shape)


	# print(x_test)
	# print(x_test.shape)
	# result = crop_images(location_test, x_test, size_x_test, size_y_test)
	# print(result)
	# print(result.shape)