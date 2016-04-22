import numpy as np
import pandas
import cv2
import itertools
import pickle
import os.path

from sklearn.preprocessing import StandardScaler

import keras_1
from keras.callbacks import EarlyStopping

from sklearn.metrics import log_loss

result_img_size = (40, 40)
color = 0
def load_data_for_img(filename, prefix='train/'):
	img = cv2.imread(prefix + filename, color)
	crop_regions = [[[0,400], [0, 400]], [[240, 640], [0, 400]], [[0,400], [80,480]], [[240, 640], [80, 480]]]
	result = [np.ndarray.flatten(cv2.resize(img[region[0][0]:region[0][1], region[1][0]:region[1][1]], result_img_size, interpolation=cv2.INTER_AREA)) for region in crop_regions]
	
	return result

	#img_small = cv2.resize(img, result_img_size, interpolation=cv2.INTER_AREA)
	#pixels = np.ndarray.flatten(img_small)

	#return [pixels]

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

def getInputsAndLabelsForSubjects(all_data, subject_indices):
	inputs = np.concatenate([all_data[0][x] for x in subject_indices]).astype('float32')
	labels = np.concatenate([all_data[1][x] for x in subject_indices])
	filenames = [filename for x in subject_indices for filename in all_data[2][x]]
	return inputs, labels, filenames

def merge_labels_by_filename(filenames, labels):
	files_labels = zip(filenames, labels)
	result = []
	for filename, names_labels in itertools.groupby(files_labels, key=lambda x: x[0]):
		labels = list(map(lambda x: x[1], names_labels))
		result.append(np.mean(list(labels), axis=0))
	return np.array(result)


all_subject_scores = {}
num_test_subjects = 5
num_valid_subjects = 1
for fold in range(int(np.ceil(len(uniqueSubjects)/(num_test_subjects + num_valid_subjects)))):
	test_indices = list(range(fold*(num_test_subjects),(fold+1)*(num_test_subjects)))
	valid_indices = list(range((fold+1)*num_test_subjects, (fold+1)*num_test_subjects+num_valid_subjects))
	train_indices = [x for x in range(len(uniqueSubjects)) if not (x in test_indices or x in valid_indices)]
	
	#print('test: %s  train: %s   valid: %s' % (test_indices, valid_indices, train_indices))	

	test_subjects = [uniqueSubjects[x] for x in test_indices]
	valid_subjects = [uniqueSubjects[x] for x in valid_indices]	
	train_subjects = [uniqueSubjects[x] for x in train_indices]

	print('train on %s' % train_subjects)
	print('test on %s' % test_subjects)
	print('validate on %s' % valid_subjects)
	
	train_inputs, train_labels, train_filenames = getInputsAndLabelsForSubjects(subjects_data, train_indices)
	valid_inputs, valid_labels, valid_filenames = getInputsAndLabelsForSubjects(subjects_data, valid_indices)
	test_inputs, test_labels, test_filenames = getInputsAndLabelsForSubjects(subjects_data, test_indices)
	
	scaler = StandardScaler(copy=False)
	train_inputs = scaler.fit_transform(train_inputs)
	valid_inputs = scaler.transform(valid_inputs, copy=False)
	test_inputs = scaler.transform(test_inputs)

	cls = keras_1.create_model_conv1(result_img_size[0], result_img_size[1], color)
	early_stop = EarlyStopping(monitor='val_loss', patience=50, verbose=1)

	cls.fit(train_inputs, 
			dense_to_one_hot(train_labels), 
			shuffle=True, 
			nb_epoch=100, 
			batch_size = 256, 
			validation_data=(valid_inputs, dense_to_one_hot(valid_labels)), 
			callbacks=[early_stop])

	test_predictions = cls.predict_proba(test_inputs)

	score = log_loss(dense_to_one_hot(test_labels), test_predictions)	
	print('naive test log loss score: %s' % score)

	for test_index in test_indices:
		test_single_inputs, test_single_labels, test_single_filenames = getInputsAndLabelsForSubjects(subjects_data, [test_index])
		test_single_inputs = scaler.transform(test_single_inputs)
		
		test_single_predictions = cls.predict_proba(test_single_inputs)
		test_single_score = log_loss(dense_to_one_hot(test_single_labels), test_single_predictions)
		all_subject_scores[uniqueSubjects[test_index]] = test_single_score
		print('Testing on single subject %s: %s' % (uniqueSubjects[test_index], test_single_score))

		merged_predictions = merge_labels_by_filename(test_single_filenames, test_single_predictions)
		merged_labels_one_hot = merge_labels_by_filename(test_single_filenames, dense_to_one_hot(test_single_labels))
		merged_subject_score = log_loss(merged_labels_one_hot, merged_predictions)
		print('merged subject score: %s' % merged_subject_score)

for subject in all_subject_scores:
	print('%s : %s' % (subject, all_subject_scores[subject]))
