import numpy as np
import os
import functools

from sklearn.cross_validation import KFold
from pretrained_vgg16 import read_and_normalize_and_shuffle_train_data, copy_selected_drivers


#cache_path_base = '/media/nick/TempDisk/state_farm/cache_data/train_folds_cache_%dx%dx%d_fold%dof%d_seed%d'
cache_path_base = 'cache_data/train_folds_cache_%dx%dx%d_fold%dof%d_seed%d'


def get_fold_folder_name(fold, n_folds, img_rows, img_cols, color_type, random_state):
	return cache_path_base % (img_rows, img_cols, color_type, fold, n_folds, random_state)

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


if __name__ == "__main__":
	create_train_split_data()


	for fold, data_provider in enumerate(driver_split_data_generator()):
		
		(X_train, Y_train, X_valid, Y_valid) = data_provider()
		print('fold %d' % (fold))
		print(X_train.shape)
		print(Y_train.shape)
		print(X_valid.shape)
		print(Y_valid.shape)
		print(np.mean(X_train.shape))

		X_train = None
		X_valid = None
