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


if __name__ == "__main__":
    encode, decode, name = get_conv_vae_model()
    print(encode)
    print(decode)
    print(name)