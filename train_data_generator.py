import numpy as np
import os
import functools

from sklearn.cross_validation import KFold
from pretrained_vgg16 import read_and_normalize_and_shuffle_train_data, copy_selected_drivers, read_and_normalize_test_data
import run_keras_cv_drivers_v2

cache_path_base = '/media/nick/TempDisk1/state_farm/cache_data/'#train_folds_cache_%dx%dx%d_fold%dof%d_seed%d'
#cache_path_base = 'cache_data/'

train_cache_path = cache_path_base + 'train_folds_cache_%dx%dx%d_fold%dof%d_seed%d'

test_cache_path = cache_path_base + 'test_folds_cache_%dx%dx%d_nfolds%d'

unlabelled_cache_path = 'cache_data/unlabelled_cache_%dx%dx%d'


def get_fold_folder_name(fold, n_folds, img_rows, img_cols, color_type, random_state):
	return train_cache_path % (img_rows, img_cols, color_type, fold, n_folds, random_state)

def save_fold_data(X_train, Y_train, X_valid, Y_valid, folder):
	np.save(folder + "/X_train.npy", X_train)
	np.save(folder + "/Y_train.npy", Y_train)
	np.save(folder + "/X_valid.npy", X_valid)
	np.save(folder + "/Y_valid.npy", Y_valid)

def load_fold_data(folder):
	X_train = np.load(folder + "/X_train.npy")
	Y_train = np.load(folder + "/Y_train.npy")
	X_valid = np.load(folder + "/X_valid.npy")
	Y_valid = np.load(folder + "/Y_valid.npy")

	return (X_train, Y_train, X_valid, Y_valid)

def create_train_split_data(n_folds = 4, img_rows = 224, img_cols = 224, color_type = 3, random_state = 30):

	train_data, train_target, driver_id, unique_drivers = read_and_normalize_and_shuffle_train_data(img_rows, img_cols,
									color_type, shuffle=False, transform=False)

	kf = KFold(len(unique_drivers), n_folds=n_folds, shuffle=True, random_state=random_state)

	for fold, (train_drivers, test_drivers) in enumerate(kf):
		folder = get_fold_folder_name(fold, n_folds, img_rows, img_cols, color_type, random_state)

		if os.path.exists(folder):
			print('already create train split cache file %s' % folder)
			continue

		os.mkdir(folder)

		unique_list_train = [unique_drivers[i] for i in train_drivers]
		X_train, Y_train, train_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)
		unique_list_valid = [unique_drivers[i] for i in test_drivers]
		X_valid, Y_valid, test_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)

		print('Split train: ', len(X_train), len(Y_train))
		print('Split valid: ', len(X_valid), len(Y_valid))
		print('Train drivers: ', unique_list_train)
		print('Test drivers: ', unique_list_valid)

		#mix up data
		perm = np.random.permutation(len(X_train))
		save_fold_data(X_train[perm], Y_train[perm], X_valid, Y_valid, folder)

		print('done caching fold')

def driver_split_data_generator(n_folds = 4, img_rows = 224, img_cols = 224, color_type = 3, random_state = 30):
	for fold in range(n_folds):
		folder = get_fold_folder_name(fold, n_folds, img_rows, img_cols, color_type, random_state)

		if not os.path.exists(folder):
			raise Exception("Data cache file doesn't exist, run create_train_split_data() first")

		result = functools.partial(load_fold_data, folder)

		yield result


def load_test_data_fold(folder, fold):
	data = np.load(folder + '/data%d.npy' % fold)
	ids = np.load(folder + "/ids%d.npy" % fold)

	return data, ids

def save_test_data_fold(folder, fold, data, ids):
	np.save(folder + "/data%d.npy" % fold, data)
	np.save(folder + "/ids%d.npy" % fold, ids)

def get_test_folder_name(n_folds, img_rows, img_cols, color_type):
	return test_cache_path % (img_rows, img_cols, color_type, n_folds)

def test_data_generator(n_folds = 5, img_rows = 224, img_cols = 224, color_type=3):
	for fold in range(n_folds):
		folder = get_test_folder_name(n_folds, img_rows, img_cols, color_type)

		if not os.path.exists(folder):
			print(folder)
			raise Exception("Test data cache file doesn't exist, run create_test_split_data() first")

		result = functools.partial(load_test_data_fold, folder, fold)

		yield result

def create_test_split_data(n_folds = 5, img_rows = 224, img_cols = 224, color_type = 3):
    n = 79726
    folder = get_test_folder_name(n_folds, img_rows, img_cols, color_type)

    for fold in range(n_folds):
        index_range = [int(fold*n/n_folds),int((fold+1)*n/n_folds)]

        split_test_data, split_ids = read_and_normalize_test_data(img_rows, img_cols, color_type, index_range = index_range, transform=False)

        save_test_data_fold(folder, fold, split_test_data, split_ids)


def load_all_unlabeled_data(folder):
	result = np.load(folder + '/data.npy')
	return result

def save_all_unlabeled_data(folder, data):
	np.save(folder + '/data.npy', data)

def get_unlabelled_folder_name(img_rows, img_cols, color_type):
	return unlabelled_cache_path % (img_rows, img_cols, color_type)

def get_unlabelled_data(img_rows = 128, img_cols = 128, color_type=3):
	folder = get_unlabelled_folder_name(img_rows, img_cols, color_type)

	data = load_all_unlabeled_data(folder)

	return data

def create_unlabelled_data(img_rows = 128, img_cols = 128, color_type=3):

	n = 79726 + 22424

	result = np.zeros((n, color_type, img_rows, img_cols), dtype='float32')

	result[:79726], _ = run_keras_cv_drivers_v2.read_and_normalize_test_data(img_rows, img_cols, color_type)
	print('got test data')

	result[79726:], train_target, train_id, driver_id, unique_drivers = run_keras_cv_drivers_v2.read_and_normalize_train_data(img_rows, img_cols, color_type)

	print('got train data')

	folder = get_unlabelled_folder_name(img_rows, img_cols, color_type)
	save_all_unlabeled_data(folder, result)

if __name__ == "__main__":

	if 1:
		create_train_split_data(n_folds=2,img_rows=128, img_cols=128)

		for fold, data_provider in enumerate(driver_split_data_generator(n_folds=2,img_rows=128, img_cols=128)):

			(X_train, Y_train, X_valid, Y_valid) = data_provider()
			print('fold %d' % (fold))
			print(X_train.shape)
			print(Y_train.shape)
			print(X_valid.shape)
			print(Y_valid.shape)
			print(np.mean(X_train.shape))

			X_train = None
			X_valid = None

	if 0:
		#create_test_split_data()

		for fold, data_provider in enumerate(test_data_generator()):
			data, ids = data_provider()
			print(fold)
			print(ids.shape)
			print(data.shape)

	if 1:
		create_unlabelled_data(128, 128, 3)

		data = get_unlabelled_data(128, 128, 3)
		print(data.shape)
