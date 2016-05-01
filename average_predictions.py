import numpy as np
from sklearn.metrics import log_loss

n=225

def dense_to_one_hot(labels_dense, num_classes=10):
	"""Convert class labels from scalars to one-hot vectors."""
	num_labels = len(labels_dense)
	index_offset = np.arange(num_labels) * num_classes
	labels_one_hot = np.zeros((num_labels, num_classes))

	#labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1.0
	for i, c in enumerate(labels_dense):
		labels_one_hot[i,c] = 1.0

	# exit(0)
	return labels_one_hot


multi_predictions = np.load('validation_predictions2.npy').reshape(-1, n, 10)
labels = np.load('validation_single_labels.npy')
labels_one_hot = dense_to_one_hot(labels)

print(multi_predictions.shape)
print(labels.shape)



#log loss on every single example
single_predictions = multi_predictions.reshape(-1, 10)
single_labels = np.repeat(labels_one_hot, n).reshape(-1, 10)
print(single_predictions.shape)
print(single_labels.shape)

score = log_loss(single_labels, single_predictions)
print('single standalone prediction score: %s' % score)


#arithmetic means
arithmetic_predictions = np.mean(multi_predictions, axis=1)
score = log_loss(labels_one_hot, arithmetic_predictions)
print('arithmetic mean prediction score: %s' % score)


#geometric means
geometric_predictions = np.exp(np.mean(np.log(multi_predictions), axis=1))
score = log_loss(labels_one_hot, geometric_predictions)
print('geometric mean prediction score: %s' % score)