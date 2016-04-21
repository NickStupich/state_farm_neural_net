import numpy as np
import pandas
import cv2
import itertools
import pickle
import os.path

from sklearn.preprocessing import StandardScaler

import keras_1
from keras.callbacks import EarlyStopping


result_img_size = (64, 48)
color = 0
def load_data_for_img(filename, prefix='train/'):
	img = cv2.imread(prefix + filename, color)
	img_small = cv2.resize(img, result_img_size, interpolation=cv2.INTER_AREA)
	pixels = np.ndarray.flatten(img_small)

	return [pixels]


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

def load_all_subject_data():
	cache_fn = 'data.pickle'

	if os.path.isfile(cache_fn):
		print('Loading data from pickled file')
		result = pickle.load(open(cache_fn, 'rb'))
	else:
		print('Loading data from individual images')
		fn = 'driver_imgs_list.csv'
		pd = pandas.read_csv(fn)

		uniqueSubjects = list(set(pd["subject"]))
		result_filenames = []
		result_labels = []
		result_data = []

		for subject in uniqueSubjects:
			rows = pd[(pd.subject == subject)]

			print('subject: %s   count: %s' % (subject, rows.shape))

			subject_data = []
			subject_labels = []
			subject_filenames = []

			for c, img_name in zip(rows.classname, rows.img):
				filename = c + '/' + img_name
				feature_maps = load_data_for_img(filename)
				label = int(c[1:])

				subject_data += feature_maps
				subject_labels += [label] * len(feature_maps)
				subject_filenames += [filename] * len(feature_maps)

			print(len(subject_data))

			result_data.append(np.array(subject_data))
			result_labels.append(np.array(subject_labels))
			result_filenames.append(subject_filenames)

		# result_data = np.array(result_data)
		# result_labels = np.array(result_labels)

		result = (result_data, result_labels, result_filenames, uniqueSubjects)
		pickle.dump(result, open(cache_fn, 'wb'))

	return result

subjects_data = load_all_subject_data()
# pickle.dump(subjects_data_dict, open(pickle_fn, 'wb'))
uniqueSubjects = subjects_data[3]
print(uniqueSubjects)
print(len(subjects_data[0]))
print(subjects_data[0][0].shape)

num_test_subjects = 4
num_valid_subjects = 1
for fold in range(int(len(uniqueSubjects)/(num_test_subjects + num_valid_subjects))):
	test_indices = list(range(fold*(num_test_subjects+num_valid_subjects),(fold+1)*(num_test_subjects+num_valid_subjects) - num_valid_subjects))
	valid_indices = list(range(fold*(num_test_subjects+num_valid_subjects)+num_test_subjects,(fold+1)*(num_test_subjects+num_valid_subjects)))
	train_indices = [x for x in range(len(uniqueSubjects)) if not (x in test_indices or x in valid_indices)]

	print(test_indices)
	test_subjects = [uniqueSubjects[x] for x in test_indices]
	valid_subjects = [uniqueSubjects[x] for x in valid_indices]	
	train_subjects = [uniqueSubjects[x] for x in train_indices]

	print('train on %s' % train_subjects)
	print('test on %s' % test_subjects)
	print('validate on %s' % valid_subjects)

	# train_data = list(itertools.chain(*map(lambda subject: subjects_data_dict[subject], train_subjects)))
	# train_inputs = np.concatenate(list(map(lambda x: x[2], train_data)))
	# train_labels = np.concatenate(list(map(lambda x: [x[0] for _ in range(len(x[2]))], train_data)))

	train_inputs = np.concatenate([subjects_data[0][x] for x in train_indices])
	train_labels = np.concatenate([subjects_data[1][x] for x in train_indices])

	valid_inputs = np.concatenate([subjects_data[0][x] for x in valid_indices])
	valid_labels = np.concatenate([subjects_data[1][x] for x in valid_indices])

	test_inputs = np.concatenate([subjects_data[0][x] for x in test_indices])
	test_labels = np.concatenate([subjects_data[1][x] for x in test_indices])
	#valid_inputs = np.concatenate(list(map(lambda x: x[2], valid_data)))
	#valid_labels = np.concatenate(list(map(lambda x: [x[0] for _ in range(len(x[2]))], valid_data)))

	#valid_inputs2 = np.ndarray.flatten(list(map(lambda subject: map(lambda x: x[2], subjects_data_dict[subject]), valid_subjects)))

	print(train_inputs.shape)
	print(valid_inputs.shape)
	print(test_inputs.shape)

	# exit(0)
	
	scaler = StandardScaler(copy=False)
	train_inputs = scaler.fit_transform(train_inputs)
	valid_inputs = scaler.transform(valid_inputs, copy=False)
	test_inputs = scaler.transform(test_inputs)

	cls = keras_1.create_model_conv1(result_img_size[0], result_img_size[1], color)
	early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

	cls.fit(train_inputs, 
			dense_to_one_hot(train_labels), 
			shuffle=True, 
			nb_epoch=200, 
			batch_size = 256, 
			validation_data=(valid_inputs, dense_to_one_hot(valid_labels)), 
			callbacks=[early_stop])

	test_predictions = cls.predict_proba(test_inputs)

	score = log_loss(dense_to_one_hot(test_labels), test_predictions)	
	print('test log loss score: %s' % score)