import theano
import theano.tensor as T
import numpy as np

if 1:
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
	offsets = T.imatrix('offsets')
	inputs = T.ftensor4('inputs')
	outputs = T.ftensor4('outputs')

	def set_example_for_offset(offset, input_tensor, output_tensor, sizes):
		input_subtensor = input_tensor[:, :, offset[0] : offset[0] + sizes[0], offset[1] : offset[1] + sizes[1]]
		result = T.set_subtensor()