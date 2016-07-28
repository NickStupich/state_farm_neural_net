from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers import Input, Dense, Lambda
from keras.layers import Flatten, Dropout, Activation, Reshape, Merge


def get_conv_vae_model(layers = [32, 64, 128, 256], conv_size=5, subsample=2, model_name='',
    img_rows = 128, img_cols = 128, color_type=3):

    conv_name = '-'.join(['%s_%s' % (layer, conv_size) for layer in layers])
    model_name = 'conv%s_subsample' % (conv_name)

    encode_layers = []

    for layer_size in layers:
        encode_layers.append(Convolution2D(layer_size, conv_size, conv_size,
                                activation='relu', border_mode='same',
                                subsample=(subsample,subsample)))

    encode_layers.append(Flatten())

    smallest_rows = int(img_rows / (subsample)**len(layers))
    smallest_cols = int(img_cols / (subsample)**len(layers))

    decode_layers = [
        Dense(layers[-1]*smallest_rows*smallest_cols),
        Reshape((layers[-1], smallest_rows, smallest_cols),),
    ]

    for layer_size in reversed(layers[:-1]):
        decode_layers.append(UpSampling2D((subsample, subsample)))
        decode_layers.append(Convolution2D(layer_size, conv_size, conv_size, activation='relu', border_mode='same'))

    decode_layers.append(UpSampling2D((subsample, subsample)))
    decode_layers.append(Convolution2D(color_type, conv_size, conv_size, activation='sigmoid', border_mode='same'))

    return encode_layers, decode_layers, model_name

def get_vgg16_conv_model(img_rows = 128, img_cols = 128, color_type=3):
        encode_layers = []
        decode_layers = []

        model_name='vgg16_autoenc'

        encode_layers.append(Convolution2D(64, 3, 3, activation='relu', border_mode='same', input_shape=(color_type,img_rows, img_cols), init='he_normal'))
        encode_layers.append(Convolution2D(64, 3, 3, activation='relu', border_mode='same', init='he_normal'))
        encode_layers.append(MaxPooling2D((2, 2), strides=(2, 2)))

        encode_layers.append(Convolution2D(128, 3, 3, activation='relu', border_mode='same', init='he_normal'))
        encode_layers.append(Convolution2D(128, 3, 3, activation='relu', border_mode='same', init='he_normal'))
        encode_layers.append(MaxPooling2D((2, 2), strides=(2, 2)))

        encode_layers.append(Convolution2D(256, 3, 3, activation='relu', border_mode='same', init='he_normal'))
        encode_layers.append(Convolution2D(256, 3, 3, activation='relu', border_mode='same', init='he_normal'))
        encode_layers.append(Convolution2D(256, 3, 3, activation='relu', border_mode='same', init='he_normal'))
        encode_layers.append(MaxPooling2D((2, 2), strides=(2, 2)))

        encode_layers.append(Convolution2D(512, 3, 3, activation='relu', border_mode='same', init='he_normal'))
        encode_layers.append(Convolution2D(512, 3, 3, activation='relu', border_mode='same', init='he_normal'))
        encode_layers.append(Convolution2D(512, 3, 3, activation='relu', border_mode='same', init='he_normal'))
        encode_layers.append(MaxPooling2D((2, 2), strides=(2, 2)))

        encode_layers.append(Convolution2D(512, 3, 3, activation='relu', border_mode='same', init='he_normal'))
        encode_layers.append(Convolution2D(512, 3, 3, activation='relu', border_mode='same', init='he_normal'))
        encode_layers.append(Convolution2D(512, 3, 3, activation='relu', border_mode='same', init='he_normal'))
        #took out the max pooling layer here
        encode_layers.append(Flatten())


        smallest_rows = int(img_rows / (2)**4)
        smallest_cols = int(img_cols / (2)**4)


        decode_layers = [
            Dense(512*smallest_rows*smallest_cols),
            Reshape((512, smallest_rows, smallest_cols),),
        ]

        #max pool layer is not here!
        decode_layers.append(Convolution2D(512, 3, 3, activation='relu', border_mode='same', init='he_normal'))
        decode_layers.append(Convolution2D(512, 3, 3, activation='relu', border_mode='same', init='he_normal'))
        decode_layers.append(Convolution2D(512, 3, 3, activation='relu', border_mode='same', init='he_normal'))


        decode_layers.append(UpSampling2D((2, 2)))
        decode_layers.append(Convolution2D(512, 3, 3, activation='relu', border_mode='same', init='he_normal'))
        decode_layers.append(Convolution2D(512, 3, 3, activation='relu', border_mode='same', init='he_normal'))
        decode_layers.append(Convolution2D(256, 3, 3, activation='relu', border_mode='same', init='he_normal'))


        decode_layers.append(UpSampling2D((2, 2)))
        decode_layers.append(Convolution2D(256, 3, 3, activation='relu', border_mode='same', init='he_normal'))
        decode_layers.append(Convolution2D(256, 3, 3, activation='relu', border_mode='same', init='he_normal'))
        decode_layers.append(Convolution2D(128, 3, 3, activation='relu', border_mode='same', init='he_normal'))

        decode_layers.append(UpSampling2D((2, 2)))
        decode_layers.append(Convolution2D(128, 3, 3, activation='relu', border_mode='same', init='he_normal'))
        decode_layers.append(Convolution2D(64, 3, 3, activation='relu', border_mode='same', init='he_normal'))

        decode_layers.append(UpSampling2D((2, 2)))
        decode_layers.append(Convolution2D(64, 3, 3, activation='relu', border_mode='same', input_shape=(color_type,img_rows, img_cols), init='he_normal'))
        decode_layers.append(Convolution2D(color_type, 3, 3, activation='sigmoid', border_mode='same', init='he_normal'))

        return encode_layers, decode_layers, model_name

if __name__ == "__main__":
    encode, decode, name = get_conv_vae_model()
    print(encode)
    print(decode)
    print(name)
