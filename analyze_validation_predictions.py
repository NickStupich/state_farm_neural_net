import numpy as np
import pylab
from sklearn import metrics


def main(filename = 'validation_preds/run_gen_vgg16_20_0.05_0.05_10.0_0.1_10.0_driverSplit_test_augment5.npy', class_filters = None, verbose = True):


	preds_labels = np.load(filename)

	preds = preds_labels[:, :10]
	labels = preds_labels[:, 10:]
	labels_dense = np.argmax(labels, axis=1)
	preds_dense = np.argmax(preds, axis=1)

	if class_filters:
		include_indices = np.in1d(labels_dense, class_filters)

		labels_dense = labels_dense[include_indices]
		labels = labels[include_indices]
		preds = preds[include_indices]
		preds_dense = preds_dense[include_indices]
	
	if verbose:	
		for power in np.linspace(0.1, 2.0, 20):	
			print('power: %f    log loss: %s' % (power, metrics.log_loss(labels, preds**power)))
	
	accuracy = metrics.accuracy_score(labels_dense, preds_dense)

	if verbose:
		print('accuracy: %s' % accuracy)

		print('confusion matrix:')

		print(metrics.confusion_matrix(labels_dense, preds_dense))

	return accuracy

def merge_pairs_preds(fn_all, fn2, classes = [8,9]):
	preds_labels = np.load(fn_all)
	preds_labels2 = np.load(fn2)


	preds = preds_labels[:, :10]
	labels = preds_labels[:, 10:]
	labels_dense = np.argmax(labels, axis=1)
	preds_dense = np.argmax(preds, axis=1)

	preds2 = preds_labels2[:, classes]

	original_classes = preds[:,classes]

	original_sums = np.sum(original_classes, axis=1)

	new_classes = original_classes * preds2

	rescale_factors = np.divide(original_sums, np.sum(new_classes, axis=1))
	print(rescale_factors.shape)
	print(new_classes.shape)

	new_classes_normalized = new_classes #np.multiply(new_classes, rescale_factors)
	for i in range(len(new_classes)):
		new_classes_normalized[i] *= rescale_factors[i]


	print(new_classes_normalized.shape)

	preds[:, classes] = new_classes_normalized


	for power in np.linspace(0.1, 2.0, 20):	print('power: %f    log loss: %s' % (power, metrics.log_loss(labels, preds**power)))
	
	accuracy = metrics.accuracy_score(labels_dense, preds_dense)


if __name__ == "__main__":
	# for i in range(10):
	# 	for j in range(i+1, 10):
	# 		classes = [i, j]
	# 		print(classes, main(class_filters = classes, verbose = False))

	main('validation_preds/run_gen_vgg16_20_0.05_0.05_10.0_0.1_10.0_driverSplit_test_augment5.npy')#, class_filters = [8, 9])
	# main('validation_preds/run_gen_vgg16_10_0.05_0.05_10.0_0.1_10.0_classes8,9_driverSplit_test_augment5.npy', class_filters = [8, 9])

	# main()

	merge_pairs_preds('validation_preds/run_gen_vgg16_20_0.05_0.05_10.0_0.1_10.0_driverSplit_test_augment5.npy',
		'validation_preds/run_gen_vgg16_10_0.05_0.05_10.0_0.1_10.0_classes8,9_driverSplit_test_augment5.npy')