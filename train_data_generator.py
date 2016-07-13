import numpy as np


from sklearn.cross_validation import KFold
import pretrained_vgg16



def driver_split_data_generator(n_folds = 4, img_rows = 224, img_cols = 224, color_type = 3):

	train_data = None
	# train_data, train_target, driver_id, unique_drivers = pretrained_vgg16.read_and_normalize_and_shuffle_train_data(img_rows, img_cols,
									# color_type, shuffle=False, transform=False)


	kf = KFold(len(unique_drivers), n_folds=nfolds, shuffle=True, random_state=random_state)
	

	for num_fold, (train_drivers, test_drivers) in enumerate(kf):

		unique_list_train = [unique_drivers[i] for i in train_drivers]
		X_train, Y_train, train_index = pretrained_vgg16.copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)
		unique_list_valid = [unique_drivers[i] for i in test_drivers]
		X_valid, Y_valid, test_index = pretrained_vgg16.copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)

		print('Split train: ', len(X_train), len(Y_train))
		print('Split valid: ', len(X_valid), len(Y_valid))
		print('Train drivers: ', unique_list_train)
		print('Test drivers: ', unique_list_valid)