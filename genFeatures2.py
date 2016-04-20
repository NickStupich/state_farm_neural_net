import numpy as np
import pandas
import cv2
import itertools

import keras_1


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
	fn = 'driver_imgs_list.csv'
	pd = pandas.read_csv(fn)

	uniqueSubjects = list(set(pd["subject"]))
	result = {}

	for subject in uniqueSubjects:
		rows = pd[(pd.subject == subject)]

		print('subject: %s   count: %s' % (subject, rows.shape))

		subject_data = []
		for c, img_name in zip(pd.classname, pd.img)[:200]:
			filename = c + '/' + img_name
			feature_maps = load_data_for_img(filename)
			label = int(c[1:])
			subject_data.append((label, filename, feature_maps))

		result[subject] = subject_data
	return result

subjects_data_dict = load_all_subject_data()
uniqueSubjects = subjects_data_dict.keys()

num_test_subjects = 4
num_valid_subjects = 1
for fold in range(len(uniqueSubjects)/(num_test_subjects + num_valid_subjects)):
	test_subjects = uniqueSubjects[fold*(num_test_subjects+num_valid_subjects):(fold+1)*(num_test_subjects+num_valid_subjects) - num_valid_subjects]
	valid_subjects = uniqueSubjects[fold*(num_test_subjects+num_valid_subjects)+num_test_subjects:(fold+1)*(num_test_subjects+num_valid_subjects)]	
	train_subjects = uniqueSubjects[:]
	for s in test_subjects + valid_subjects : train_subjects.remove(s)

	print('train on %s' % train_subjects)
	print('test on %s' % test_subjects)
	print('validate on %s' % valid_subjects)

	train_data = list(itertools.chain(*map(lambda subject: subjects_data_dict[subject], train_subjects)))
	train_inputs = np.concatenate(map(lambda x: x[2], train_data))
	train_labels = np.concatenate(map(lambda x: [x[0] for _ in range(len(x[2]))], train_data))

	valid_data = list(itertools.chain(*map(lambda subject: subjects_data_dict[subject], valid_subjects)))
	valid_inputs = np.concatenate(map(lambda x: x[2], valid_data))
	valid_labels = np.concatenate(map(lambda x: [x[0] for _ in range(len(x[2]))], valid_data))

	test_data = list(itertools.chain(*map(lambda subject: subjects_data_dict[subject], test_subjects)))





	cls = keras_1.create_model_conv1(result_img_size[0], result_img_size[1], color)
	early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
	
	cls.fit(train_data, 
			dense_to_one_hot(train_labels), 
			shuffle=True, 
			nb_epoch=200, 
			batch_size = 256, 
			validation_data=(valid_inputs, dense_to_one_hot(valid_labels)), 
			callbacks=[early_stop])