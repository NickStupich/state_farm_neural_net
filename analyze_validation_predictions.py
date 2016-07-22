import numpy as np
import pylab
from sklearn import metrics


def main():

	fn = 'validation_preds/run_gen_vgg16_20_0.05_0.05_10.0_0.1_10.0_driverSplit_test_augment5.npy'

	preds_labels = np.load(fn)

	preds = preds_labels[:, :10]
	labels = preds_labels[:, 10:]

	for power in np.linspace(0.1, 2.0, 20):
		print('power: %f    log loss: %s' % (power, metrics.log_loss(labels, preds**power)))

	print('accuracy: %s' % metrics.accuracy_score(np.argmax(labels, axis=1), np.argmax(preds, axis=1)))



if __name__ == "__main__":
	main()