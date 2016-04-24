import numpy as np
import pandas
import cv2
import itertools
import pickle
import os
import os.path

from sklearn.preprocessing import StandardScaler

import keras_1
from keras.callbacks import EarlyStopping

from sklearn.metrics import log_loss

result_img_size = (40, 40)
color = 0
def load_data_for_img(filename, prefix='train/'):
	img = cv2.imread(prefix + filename, color)
	#crop_regions = [[[0,400], [0, 400]], [[240, 640], [0, 400]], [[0,400], [80,480]], [[240, 640], [80, 480]]]
	x_crop_size = 400
	y_crop_size = 400
	crop_regions = []
	for y_start in [0, 40, 80]:
		for x_start in [0, 80, 160, 240]:
			crop_regions.append(((x_start, x_start + x_crop_size),(y_start, y_start + y_crop_size)))
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

		result = (result_data, result_labels, result_filenames, uniqueSubjects)
		pickle.dump(result, open(cache_fn, 'wb'))

	return result

def load_testing_data():	
	cache_fn = 'test_data.pickle'

	if os.path.isfile(cache_fn):
		print('Loading test data from pickled file')
		result = pickle.load(open(cache_fn, 'rb'))
	else:
		result_data = []
		filenames = []
		all_filenames = os.listdir('test')
		for i, fn in enumerate(all_filenames):
			data = load_data_for_img(fn, 'test/')
			filenames.append([fn] * len(data))
			result_data.append(data)

			if i % 1000 == 0:
				print('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b%d/%d' % (i, len(all_filenames)))

		result = (result_data, filenames)
		pickle.dump(result, open(cache_fn, 'wb'))

	return result

def getInputsAndLabelsForSubjects(all_data, subject_indices):
	inputs = np.concatenate([all_data[0][x] for x in subject_indices]).astype('float32')
	labels = np.concatenate([all_data[1][x] for x in subject_indices])
	filenames = [filename for x in subject_indices for filename in all_data[2][x]]
	return inputs, labels, filenames

def merge_labels_by_filename(filenames, labels):
	files_labels = zip(filenames, labels)
	result = []
	filenames = []
	for filename, names_labels in itertools.groupby(files_labels, key=lambda x: x[0]):
		labels = list(map(lambda x: x[1], names_labels))
		result.append(np.mean(list(labels), axis=0))
		filenames.append(filename)
	return np.array(result), filenames

def get_trained_classifier_and_scaler(subjects_data, uniqueSubjects, train_indices, valid_indices):
	train_subjects = [uniqueSubjects[x] for x in train_indices]
	print('train on %s' % train_subjects)
	train_inputs, train_labels, train_filenames = getInputsAndLabelsForSubjects(subjects_data, train_indices)

	scaler = StandardScaler(copy=False)
	train_inputs = scaler.fit_transform(train_inputs)

	cls = keras_1.create_model_conv1(result_img_size[0], result_img_size[1], color)
	callbacks = []
	if len(valid_indices) > 0:

		valid_subjects = [uniqueSubjects[x] for x in valid_indices]
		print('validate on %s' % valid_subjects)
		valid_inputs, valid_labels, valid_filenames = getInputsAndLabelsForSubjects(subjects_data, valid_indices)
		valid_inputs = scaler.transform(valid_inputs, copy=False)
	
		early_stop = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
		callbacks.append(early_stop)


	cls.fit(train_inputs, 
			dense_to_one_hot(train_labels), 
			shuffle=True, 
			nb_epoch=20, 
			batch_size = 256, 
			validation_data=(valid_inputs, dense_to_one_hot(valid_labels)) if len(valid_indices) > 0 else None, 
			callbacks=callbacks)

	return scaler, cls

def write_submission_file(predictions, filenames):
	print('writing submission file')
	f = open('submission.csv', 'w')
	f.write('img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n')

	for fn, probs in zip(filenames, predictions):
		f.write('%s,' % fn + ','.join(map(str, probs)) + '\n')

	f.close()
	print('done submission file')

def main():	
	subjects_data = load_all_subject_data()
	uniqueSubjects = subjects_data[3]
	print(uniqueSubjects)

	if True:
		all_subject_scores = {}
		num_test_subjects = 6
		num_valid_subjects = 0

		all_predictions = []
		all_labels = []
		for fold in range(int(np.ceil(len(uniqueSubjects)/(num_test_subjects + num_valid_subjects)))):
			test_indices = list(map(lambda x: x % len(uniqueSubjects), range(fold*(num_test_subjects),(fold+1)*(num_test_subjects))))
			valid_indices = list(map(lambda x: x % len(uniqueSubjects), range((fold+1)*num_test_subjects, (fold+1)*num_test_subjects+num_valid_subjects)))
			train_indices = [x for x in range(len(uniqueSubjects)) if not (x in test_indices or x in valid_indices)]
			
			test_subjects = [uniqueSubjects[x] for x in test_indices]	
			print('test on %s' % test_subjects)	
			test_inputs, test_labels, test_filenames = getInputsAndLabelsForSubjects(subjects_data, test_indices)	
			
			scaler, cls = get_trained_classifier_and_scaler(subjects_data, uniqueSubjects, train_indices, valid_indices)
			scaled_test_inputs = scaler.transform(test_inputs)
			test_predictions = cls.predict_proba(scaled_test_inputs)

			score = log_loss(dense_to_one_hot(test_labels), test_predictions)	
			print('naive test log loss score: %s' % score)

			for test_index in test_indices:
				test_single_inputs, test_single_labels, test_single_filenames = getInputsAndLabelsForSubjects(subjects_data, [test_index])
				test_single_inputs = scaler.transform(test_single_inputs)
				
				test_single_predictions = cls.predict_proba(test_single_inputs)
				test_single_score = log_loss(dense_to_one_hot(test_single_labels), test_single_predictions)
				print('Testing on single subject %s: %s' % (uniqueSubjects[test_index], test_single_score))

				merged_predictions, merged_filenames = merge_labels_by_filename(test_single_filenames, test_single_predictions)
				merged_labels_one_hot, merged_label_filenames = merge_labels_by_filename(test_single_filenames, dense_to_one_hot(test_single_labels))
				merged_subject_score = log_loss(merged_labels_one_hot, merged_predictions)
				print('merged subject score: %s' % merged_subject_score)

				all_subject_scores[uniqueSubjects[test_index]] = merged_subject_score

				all_predictions.append(merged_predictions)
				all_labels.append(merged_labels_one_hot)

		for subject in all_subject_scores:
			print('%s : %s' % (subject, all_subject_scores[subject]))

		all_labels = np.concatenate(all_labels)
		all_predictions = np.concatenate(all_predictions)

		overall_score = log_loss(all_labels, all_predictions)
		print('overall log loss score: %s' % overall_score)


	### Make predictions on kaggle test data ###
	if True:
		train_indices = range(len(uniqueSubjects))
		scaler, cls = get_trained_classifier_and_scaler(subjects_data, uniqueSubjects, train_indices, [])

		test_data, test_filenames = load_testing_data()
		flattened_filenames = list(itertools.chain(*test_filenames))
		scaled_test_data = scaler.transform(np.concatenate(test_data).astype('float32'))

		print('length of flatten filenames: %s' % len(flattened_filenames))
		print('shape of scaled test data: %s' % str(scaled_test_data.shape))

		test_predictions = cls.predict_proba(scaled_test_data)
		merged_test_predictions, merged_test_filenames = merge_labels_by_filename(flattened_filenames, test_predictions)

		write_submission_file(merged_test_predictions, merged_test_filenames)


if __name__ == "__main__":
	main()