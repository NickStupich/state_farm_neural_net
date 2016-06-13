from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Flatten, Dropout, Activation, ZeroPadding2D
from keras.optimizers import *
from keras.models import Model, Sequential

import h5py

def get_trained_vgg16_model(img_rows, img_cols, color_type):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(color_type,img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    model.load_weights('vgg16_weights.h5')

    return model

def set_vgg16_model_2_weights(model, set_last_layer = True):

    f = h5py.File('vgg16_weights.h5')
    model_k = 0
    model_num_layers = len(model.layers) if set_last_layer else (len(model.layers)-1)

    for k in range(f.attrs['nb_layers']):
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        if len(weights) > 0:
            while model_k < model_num_layers and len(model.layers[model_k].get_weights()) == 0:
                model_k += 1
                #print('skipping model layer %d' % model_k)

            #print('setting weights from full model layer %d to layer %d' % (k, model_k))
            if model_k == model_num_layers: break
            #print('setting layer %d / %d' % (model_k, model_num_layers))
            model.layers[model_k].set_weights(weights)
            model_k += 1
    f.close()


def get_trained_vgg16_model_2(img_rows, img_cols, color_type, output_size = 1000):
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', input_shape=(color_type,img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_size, activation='softmax'))

    set_last_layer = (output_size == 1000)
    set_vgg16_model_2_weights(model, set_last_layer = set_last_layer)

    return model

def main():
    model1 = get_trained_vgg16_model(224, 224, 3)
    model1.compile('sgd', 'mse')

    model2 = get_trained_vgg16_model_2(224, 224, 3)
    model2.compile('sgd', 'mse')

    n = 10
    test_input = np.random.uniform(low=0.0, high=1.0, size=(n, 3, 224, 224))
    batch_size = 1
    encode1 = model1.predict(test_input, verbose=True, batch_size = batch_size)
    encode2 = model2.predict(test_input, verbose=True, batch_size = batch_size)

    print(encode1)
    print(encode2)
    diff = encode1 - encode2
    print(np.sum(diff))

if __name__ == "__main__":
    main()