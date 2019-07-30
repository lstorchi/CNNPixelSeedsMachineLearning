import argparse
import dataset
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Dropout, BatchNormalization
from keras.models import Model
from keras import optimizers
from keras.constraints import max_norm
from keras.utils import plot_model

IMAGE_SIZE = dataset.padshape


def adam_small_doublet_model(args, n_channels,n_infos):
    hit_shapes = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, n_channels), name='hit_shape_input')
    
    infos = Input(shape=(n_infos,), name='info_input')

    drop = Dropout(args.dropout)(hit_shapes)
    conv = Conv2D(32, (5, 5), activation='relu', padding='same', data_format="channels_last", name='conv1')(drop)
    conv = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last", name='conv2')(conv)
    pool = MaxPooling2D(pool_size=(2, 2), padding='same', data_format="channels_last", name='pool1')(conv)

    conv = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last", name='conv3')(pool)
    conv = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last", name='conv4')(conv)
    pool = MaxPooling2D(pool_size=(2, 2), padding='same', data_format="channels_last", name='pool2')(conv)

    flat = Flatten()(pool)
    concat = concatenate([flat, infos])

    b_norm = BatchNormalization()(concat)
    dense = Dense(128, activation='relu', kernel_constraint=max_norm(args.maxnorm), name='dense1')(b_norm)
    drop = Dropout(args.dropout)(dense)
    dense = Dense(64, activation='relu', kernel_constraint=max_norm(args.maxnorm), name='dense2')(drop)
    drop = Dropout(args.dropout)(dense)
    pred = Dense(2, activation='softmax', kernel_constraint=max_norm(args.maxnorm), name='output')(drop)

    model = Model(inputs=[hit_shapes, infos], outputs=pred)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def big_filters_model(args, n_channels,n_infos):
    hit_shapes = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, n_channels), name='hit_shape_input')
    infos = Input(shape=(n_infos,), name='info_input')

    conv = Conv2D(128, (5, 5), activation='relu', padding='valid', data_format="channels_last", name='conv1')(hit_shapes)

    flat = Flatten()(conv)
    concat = concatenate([flat, infos])

    drop = Dropout(args.dropout)(concat)
    dense = Dense(256, activation='relu', kernel_constraint=max_norm(args.maxnorm), name='dense1')(drop)
    drop = Dropout(args.dropout)(dense)
    dense = Dense(64, activation='relu', kernel_constraint=max_norm(args.maxnorm), name='dense2')(drop)
    drop = Dropout(args.dropout)(dense)
    pred = Dense(2, activation='softmax', kernel_constraint=max_norm(args.maxnorm), name='output')(drop)

    model = Model(inputs=[hit_shapes, infos], outputs=pred)
    my_sgd = optimizers.SGD(lr=args.lr, decay=1e-5, momentum=args.momentum, nesterov=True)
    model.compile(optimizer=my_sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def dense_model(args, n_channels,n_infos):
    hit_shapes = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, n_channels), name='hit_shape_input')
    infos = Input(shape=(n_infos,), name='info_input')
    flat = Flatten()(hit_shapes)
    concat = concatenate([flat, infos])

    b_norm = BatchNormalization()(concat)
    dense = Dense(256, activation='relu', kernel_constraint=max_norm(args.maxnorm), name='dense1')(b_norm)
    drop = Dropout(args.dropout)(dense)
    dense = Dense(128, activation='relu', kernel_constraint=max_norm(args.maxnorm), name='dense2')(drop)
    drop = Dropout(args.dropout)(dense)
    dense = Dense(64, activation='relu', kernel_constraint=max_norm(args.maxnorm), name='dense3')(drop)
    drop = Dropout(args.dropout)(dense)
    pred = Dense(2, activation='softmax', kernel_constraint=max_norm(args.maxnorm), name='output')(drop)

    model = Model(inputs=[hit_shapes, infos], outputs=pred)
    my_sgd = optimizers.SGD(lr=args.lr, decay=1e-4, momentum=args.momentum, nesterov=True)
    model.compile(optimizer=my_sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def small_doublet_model(args, n_channels,n_infos):
    hit_shapes = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, n_channels), name='hit_shape_input')
    
    infos = Input(shape=(n_infos,), name='info_input')

    drop = Dropout(args.dropout)(hit_shapes)
    conv = Conv2D(32, (5, 5), activation='relu', padding='same', data_format="channels_last", name='conv1')(drop)
    conv = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last", name='conv2')(conv)
    pool = MaxPooling2D(pool_size=(2, 2), padding='same', data_format="channels_last", name='pool1')(conv)

    conv = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last", name='conv3')(pool)
    conv = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last", name='conv4')(conv)
    pool = MaxPooling2D(pool_size=(2, 2), padding='same', data_format="channels_last", name='pool2')(conv)

    flat = Flatten()(pool)
    concat = concatenate([flat, infos])

    b_norm = BatchNormalization()(concat)
    dense = Dense(128, activation='relu', kernel_constraint=max_norm(args.maxnorm), name='dense1')(b_norm)
    drop = Dropout(args.dropout)(dense)
    dense = Dense(64, activation='relu', kernel_constraint=max_norm(args.maxnorm), name='dense2')(drop)
    drop = Dropout(args.dropout)(dense)
    pred = Dense(2, activation='softmax', kernel_constraint=max_norm(args.maxnorm), name='output')(drop)

    model = Model(inputs=[hit_shapes, infos], outputs=pred)
    my_sgd = optimizers.SGD(lr=args.lr, decay=1e-4, momentum=args.momentum, nesterov=True)
    model.compile(optimizer=my_sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def big_doublet_model(args, n_channels,n_infos):
    hit_shapes = Input(shape=(8, 8, n_channels,n_infos), name='hit_shape_input')
    infos = Input(shape=(n_infos,), name='info_input')

    drop = Dropout(args.dropout)(hit_shapes)
    conv = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last", name='conv1')(drop)
    conv = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last", name='conv2')(conv)
    pool = MaxPooling2D(pool_size=(2, 2), padding='same', data_format="channels_last", name='pool1')(conv)

    conv = Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_last", name='conv3')(pool)
    conv = Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_last", name='conv4')(conv)
    pool = MaxPooling2D(pool_size=(2, 2), padding='same', data_format="channels_last", name='pool2')(conv)

    flat = Flatten()(pool)
    concat = concatenate([flat, infos])

    drop = Dropout(args.dropout)(concat)
    dense = Dense(256, activation='relu', kernel_constraint=max_norm(args.maxnorm), name='dense1')(drop)
    drop = Dropout(args.dropout)(dense)
    dense = Dense(64, activation='relu', kernel_constraint=max_norm(args.maxnorm), name='dense2')(drop)
    drop = Dropout(args.dropout)(dense)
    pred = Dense(2, activation='softmax', kernel_constraint=max_norm(args.maxnorm), name='output')(drop)

    model = Model(inputs=[hit_shapes, infos], outputs=pred)
    my_sgd = optimizers.SGD(lr=args.lr, decay=1e-4, momentum=args.momentum, nesterov=True)
    model.compile(optimizer=my_sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def separate_conv_doublet_model(args, n_channels,n_infos):
    in_hit_shapes = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, n_channels), name='in_hit_shape_input')
    out_hit_shapes = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, n_channels), name='out_hit_shape_input')
    infos = Input(shape=(n_infos,), name='info_input')

    # input shape convolution
    drop = Dropout(args.dropout)(in_hit_shapes)
    conv = Conv2D(64, (5, 5), activation='relu', padding='same', data_format="channels_last", name='in_conv1')(drop)
    conv = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last", name='in_conv2')(conv)
    pool = MaxPooling2D(pool_size=(2, 2), padding='same', data_format="channels_last", name='in_pool1')(conv)

    conv = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last", name='in_conv3')(pool)
    conv = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last", name='in_conv4')(conv)
    pool = MaxPooling2D(pool_size=(2, 2), padding='same', data_format="channels_last", name='in_pool2')(conv)
    in_flat = Flatten()(pool)

    # output shape convolution
    drop = Dropout(args.dropout)(out_hit_shapes)
    conv = Conv2D(64, (5, 5), activation='relu', padding='same', data_format="channels_last", name='out_conv1')(drop)
    conv = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last", name='out_conv2')(conv)
    pool = MaxPooling2D(pool_size=(2, 2), padding='same', data_format="channels_last", name='out_pool1')(conv)

    conv = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last", name='out_conv3')(pool)
    conv = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last", name='out_conv4')(conv)
    pool = MaxPooling2D(pool_size=(2, 2), padding='same', data_format="channels_last", name='out_pool2')(conv)
    out_flat = Flatten()(pool)

    concat = concatenate([in_flat, out_flat, infos])
    info_drop = Dropout(args.dropout)(concat)

    dense = Dense(256, activation='relu', kernel_constraint=max_norm(args.maxnorm), name='dense1')(info_drop)
    drop = Dropout(args.dropout)(dense)
    dense = Dense(64, activation='relu', kernel_constraint=max_norm(args.maxnorm), name='dense2')(drop)
    drop = Dropout(args.dropout)(dense)
    pred = Dense(2, activation='softmax', kernel_constraint=max_norm(args.maxnorm), name='output')(drop)

    model = Model(inputs=[in_hit_shapes, out_hit_shapes, infos], outputs=pred)
        
    model.compile(optimizer=my_sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--log_dir', type=str, default="models/cnn_doublet")
    parser.add_argument('--name', type=str, default='model_')
    parser.add_argument('--maxnorm', type=float, default=10.)
    parser.add_argument('--verbose', type=int, default=1)
    main_args = parser.parse_args()

    plot_model(big_doublet_model(main_args, 8), to_file='big_model.png', show_shapes=True, show_layer_names=True)
    plot_model(separate_conv_doublet_model(main_args, 4), to_file='separate_conv_model.png', show_shapes=True, show_layer_names=True)
