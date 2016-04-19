import numpy as np
import pandas
import random
import pylab


from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from keras.callbacks import EarlyStopping

import keras_1

img_cols = 64
img_rows = 48

isColor = 1

def load_driver_ids():
	fn = 'driver_imgs_list.csv'
	pd = pandas.read_csv(fn)
	img_fns = np.asarray(pd['img'])
	subjects = list(pd['subject'])

	# print subjects[:50]

	img_nums = list(map(lambda s: int(s.split('_')[1].split('.')[0]), img_fns))

	# print 'img_nums: ', img_nums[:50]
	return img_nums, subjects

def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))

  #labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1.0
  for i, c in enumerate(labels_dense):
  	labels_one_hot[i,c] = 1.0

  # exit(0)
  return labels_one_hot

def getPredictions(train_data, train_labels, test_data, test_labels, isColor):

	train_mean = np.mean(train_data)
	train_std = np.std(train_data)
	# train_std = 10

	print('mean: %s, std: %s ' % (train_mean, train_std))

	train_data = (train_data - train_mean) / train_std
	test_data = (test_data - train_mean) / train_std
	# print train_data

	# cls = LogisticRegression(C = 1E-4)
	# cls = keras_1.create_model_v1(64, 48)
	#cls = keras_1.create_model_logReg(64, 48)
	cls = keras_1.create_model_conv1(img_cols, img_rows, isColor = isColor)
	# cls = LinearDiscriminantAnalysis()
	# cls = RandomForestClassifier()

	# shuffled_indices = np.random.permutation(len(train_data))
	# shuffled_train_data = train_data[shuffled_indices]
	# shuffled_train_labels = train_labels[shuffled_indices]

	print('training... data shape: %s ' % str(train_data.shape))
	early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
	# cls.fit(train_data, dense_to_one_hot(train_labels), shuffle=True, nb_epoch=200, batch_size = 256, validation_split=0.05, callbacks=[early_stop])

	n_test = len(test_data)
	all_indices = np.arange(n_test)
	np.random.shuffle(all_indices)

	test_validation_indices = all_indices[:n_test / 2]
	test_test_indices = all_indices[n_test/2:]
	test_validation_data = test_data[test_validation_indices,:]
	test_test_data = test_data[test_test_indices,:]
	test_validation_output = test_labels[test_validation_indices]
	print(test_validation_data.shape)
	cls.fit(train_data, 
			dense_to_one_hot(train_labels), 
			shuffle=True, 
			nb_epoch=200, 
			batch_size = 256, 
			validation_data=(test_validation_data, dense_to_one_hot(test_validation_output)), 
			callbacks=[early_stop])

	print('done training')

	test_predictions = np.array(cls.predict_proba(test_test_data))

	train_predictions = cls.predict_proba(train_data)
	print('training log loss: %s' % log_loss(dense_to_one_hot(train_labels), train_predictions))

	all_predictions = np.array(cls.predict_proba(test_data))

	return all_predictions

def main():
	img_nums, subjects = load_driver_ids()

	data = np.load('downsampled_(%d, %d)_%d.npy' % (img_cols, img_rows, isColor))
	
	data_class = data[:,0]
	data_file_num = data[:,1]
	data_pixels = data[:,2:]


	data_indices_map = dict((file_num, i) for i, file_num in enumerate(data_file_num))

	unique_subjects = list(set(subjects))

	all_predictions = []
	all_outputs = []

	n = 4
	for subject in [unique_subjects[i*n:(i+1)*n] for i in range(int(len(unique_subjects)/n))]:
		print('subjects: %s' % subject)

		test_indices = np.array(list(map(lambda b: b[1], filter(lambda a: a[0] in subject, zip(subjects, img_nums)))))
		train_indices = np.array(list(map(lambda b: b[1], filter(lambda a: not a[0] in subject, zip(subjects, img_nums)))))
		
		print('test count: %d    train count: %d' % (len(test_indices), len(train_indices)))

		data_test_indices = np.array([data_indices_map[i] for i in test_indices])
		data_train_indices = np.array([data_indices_map[i] for i in train_indices])

		test_classes = data_class[data_test_indices]
		train_classes = data_class[data_train_indices]

		test_output = dense_to_one_hot(test_classes)
		train_output = dense_to_one_hot(train_classes)

		test_data = data_pixels[data_test_indices]
		train_data = data_pixels[data_train_indices]

		predictions = getPredictions(train_data, train_classes, test_data, test_classes, isColor = isColor)

		all_predictions.append(predictions)
		all_outputs.append(test_output)

		score = log_loss(test_output, predictions)
		print('subject %s score: %s' % (subject, score))


	all_outputs = np.concatenate(all_outputs)
	all_predictions = np.concatenate(all_predictions)

	print('overall score: %s' % log_loss(all_outputs, all_predictions))

if __name__ == "__main__":
	main()
