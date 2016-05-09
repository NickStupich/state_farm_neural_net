import numpy as np
np.random.seed(1)

from keras import models
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.core import Dense, Activation, Dropout, Flatten, Reshape

from keras_convautoencoder.autoencoder_layers import DependentDense, Deconvolution2D, DePool2D
from keras_convautoencoder.helpers import show_representations2

from croppedInNet import load_all_subject_data, getInputsAndLabelsForSubjects

# def load_data():
#     (X_train, y_train), (X_test, y_test) = mnist.load_data()
#     X_train = X_train.astype("float32") / 255.0
#     X_test = X_test.astype("float32") / 255.0
#     X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]))
#     X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]))
#     X_train -= np.mean(X_train)
#     X_test -= np.mean(X_test)
#     return (X_train, y_train), (X_test, y_test)


result_img_size = (64, 48)

def build_model(nb_filters=16, nb_pool=2, nb_conv=5, img_rows = 64, img_cols = 48, color_dim = 1):
    model = models.Sequential()
    #d = Dense(100)
    c = Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same', input_shape=(1, 64, 48))
    # c2 = Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same', input_shape=(1, 64, 48))
    
    mp = MaxPooling2D(pool_size=(nb_pool, nb_pool))
    # mp2 = MaxPooling2D(pool_size=(nb_pool, nb_pool))
    
    model.add(Reshape(input_shape=(img_rows*img_cols*color_dim,), target_shape = (color_dim, img_rows, img_cols)))
    
    # =========      ENCODER     ========================
    model.add(c)
    model.add(Activation('relu'))
    model.add(mp)

    # model.add(c2)
    # model.add(Activation('tanh'))
    # model.add(mp2)

    
    # =========      BOTTLENECK     ======================
    # model.add(Flatten())
    # model.add(d)
    # model.add(Activation('tanh'))
    
    #model.add(Dropout(0.25))

    # # =========      BOTTLENECK^-1   =====================
    # model.add(DependentDense(nb_filters * 16 * 12, d))
    # model.add(Activation('tanh'))
    # model.add(Reshape((nb_filters, 16, 12)))
    # =========      DECODER     =========================

    # model.add(DePool2D(mp2, size=(nb_pool, nb_pool)))
    # model.add(Deconvolution2D(c2, border_mode='same'))
    
    # model.add(Activation('tanh'))

    model.add(DePool2D(mp, size=(nb_pool, nb_pool)))
    model.add(Deconvolution2D(c, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Flatten())

    return model


if __name__ == '__main__':
    # (X_train, y_train), (X_test, y_test) = load_data()


    subjects_data = load_all_subject_data()

    num_test_subjects = 5
    num_valid_subjects = 0

    uniqueSubjects = subjects_data[3]
    num_test_subjects = 5

    num_folds = int(np.ceil(len(uniqueSubjects)/(num_test_subjects + num_valid_subjects)))
    num_folds = 1

    for fold in range(num_folds):
        test_indices = list(map(lambda x: x % len(uniqueSubjects), range(fold*(num_test_subjects),(fold+1)*(num_test_subjects))))
        valid_indices = []
        train_indices = [x for x in range(len(uniqueSubjects)) if not (x in test_indices or x in valid_indices)]
        
        test_inputs, test_labels, test_filenames = getInputsAndLabelsForSubjects(subjects_data, test_indices)   
        train_inputs, train_labels, train_filenames = getInputsAndLabelsForSubjects(subjects_data, train_indices)   

        print(train_inputs.shape)
        print(train_labels.shape)

        train_inputs /= 255
        test_inputs /= 255

        print(train_inputs.shape)

        # exit(0)
            
        model = build_model()
        if True:
            model.compile(optimizer='rmsprop', loss='mean_squared_error')
            model.summary()
            callbacks=[]
            # callbacks.append(EarlyStopping(patience=3))
            model.fit(train_inputs, train_inputs, nb_epoch=50, batch_size=512,
                      callbacks=callbacks)
            model.save_weights('./conv.neuro', overwrite=True)
        else:
            model.load_weights('./conv.neuro')
            model.compile(optimizer='rmsprop', loss='mean_squared_error')

        show_representations2(model, test_inputs)
