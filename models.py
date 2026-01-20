import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,Reshape, Flatten, Conv1D, Conv2D, MaxPooling2D, BatchNormalization, Resizing, GlobalAveragePooling2D, GlobalMaxPool2D, Maximum, Minimum, Multiply, Conv2DTranspose, Concatenate, DepthwiseConv1D
import numpy as np

from functools import partial


def clone_and_change_input(model, shape):
    model_clone =  tf.keras.models.clone_model(model, input_tensors=tf.keras.layers.Input(shape=shape))
    for i, layer in enumerate(model.layers):
        try:
            weights = layer.get_weights()
            model_clone.layers[i].set_weights(weights)
        except Exception:
            print(Exception)
    
    return model_clone

class adaptateur(tf.keras.Model):

    def __init__(self, n=1, **kwargs):
        super(adaptateur, self).__init__(**kwargs)

        self.gap = GlobalAveragePooling2D()
        self.dense = Dense(n, activation='sigmoid', bias_initializer=None, use_bias=False)
        self.n = n

    def get_weights(self):
        return self.dense.weights[0].numpy().reshape(-1)

    def get_config(self):
        config = super().get_config()
        config.update({'n': self.n})
        return config
    
    def call(self, x):

        x = self.gap(x)
        x = self.dense(x)

        return x
    

class adaptateur2h(tf.keras.Model):
    def __init__(self, n=1, **kwargs):
        super(adaptateur2h, self).__init__(**kwargs)

        self.gap = GlobalAveragePooling2D()
        self.max = GlobalMaxPool2D()
        self.mult = Multiply()
        self.dense = Dense(n, activation='sigmoid', bias_initializer=None, use_bias=False)
        self.n = n

    def get_weights(self):
        return self.dense.weights[0].numpy().reshape(-1)

    def get_config(self):
        config = super().get_config()
        config.update({'n': self.n})
        return config
    
    def call(self, x):

        x1 = self.gap(x)
        x2 = self.max(x)
        x = self.mult([x1, x2])
        x = self.dense(x)

        return x


class adaptateur2h_min(tf.keras.Model):
    def __init__(self, n=1, **kwargs):
        super(adaptateur2h_min, self).__init__(**kwargs)

        self.gap = GlobalAveragePooling2D()
        self.max = GlobalMaxPool2D()
        self.min = Minimum()
        self.dense = Dense(n, activation='sigmoid', bias_initializer=None, use_bias=False)
        self.n = n

    def get_weights(self):
        return self.dense.weights[0].numpy().reshape(-1)

    def get_config(self):
        config = super().get_config()
        config.update({'n': self.n})
        return config
    
    def call(self, x):

        x1 = self.gap(x)
        x2 = self.max(x)
        x1_mult = self.dense(x1)
        x2_mult = self.dense(x2)
        x = self.min([x1_mult, x2_mult])

        return x


# Source: https://stackoverflow.com/questions/60644157/bias-only-layer-in-keras
class BiasLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(BiasLayer, self).__init__(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        return config
    
    def build(self, input_shape):
        self.bias = self.add_weight(shape=input_shape[1:],
                                    initializer='zeros',
                                    trainable=True)
    def call(self, x):
        return x + self.bias

class adaptateur2h_min_bias(tf.keras.Model):
    def __init__(self, n=1, **kwargs):
        super(adaptateur2h_min_bias, self).__init__(**kwargs)

        self.gap = GlobalAveragePooling2D()
        self.max = GlobalMaxPool2D()
        self.min = Minimum()
        self.bias = BiasLayer()
        self.dense = Dense(n, activation='sigmoid', bias_initializer=None, use_bias=False)
        self.n = n

    def get_weights(self):
        return self.dense.weights[0].numpy().reshape(-1)

    def get_config(self):
        config = super().get_config()
        config.update({'n': self.n})
        return config
    
    def call(self, x):

        x_gap = self.gap(x)
        x_max = self.max(x)
        
        x_gap = self.dense(x_gap)
        x_max = self.dense(x_max)
        x = self.min([x_gap, self.bias(x_max)])

        return x


# Residual layers

DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=5, strides=1,
                        padding="same", kernel_initializer="he_normal",
                        use_bias=False)

@keras.saving.register_keras_serializable()
class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, kernel_size=5, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.filters = filters
        self.strides = strides
        self.kernel_size = kernel_size
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters, kernel_size=kernel_size),
            tf.keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                tf.keras.layers.BatchNormalization()
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "activation": self.activation,
            "filters": self.filters,
            "strides": self.strides,
            "kernel_size": self.kernel_size
        })
        return config   
    
@keras.saving.register_keras_serializable()
class SEBlock(tf.keras.layers.Layer):
    def __init__(self, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.squeeze = tf.keras.layers.GlobalAveragePooling2D()
        self.dense_1 = tf.keras.layers.Dense(num_channels // self.ratio, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(num_channels, activation='sigmoid')

    def call(self, inputs):
        x = self.squeeze(inputs)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = tf.keras.layers.Reshape((1, 1, -1))(x)
        return inputs * x

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config

@keras.saving.register_keras_serializable()
class SE_ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", se_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.filters = filters
        self.strides = strides
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            tf.keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                tf.keras.layers.BatchNormalization()
            ]
        self.se = SEBlock(ratio=se_ratio)

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        Z = self.se(Z)  # Appliquer le bloc SE ici.
        return self.activation(Z + skip_Z)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "activation": self.activation,
            "filters": self.filters,
            "strides": self.strides,
            "se_ratio": self.se.ratio
        })
        return config


def get_resnet(filters_sequence = [[64, 3, 5], [128, 4, 5], [256, 6, 5], [512, 3, 5]], input_shape=[227, 227, 1], first_kernel_size=7, se_blocks=False):

    resnet = tf.keras.Sequential([
        DefaultConv2D(64, kernel_size=first_kernel_size, strides=2, input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
    ])

    filters_list = [e for sublist in [[(f, kernel_size)]*n for f, n, kernel_size in filters_sequence] for e in sublist]
    prev_filters = filters_list[0]

    for filters, kernel_size in filters_list:
        strides = 1 if filters == prev_filters else 2
        if se_blocks:
            resnet.add(SE_ResidualUnit(filters, strides=strides))
        else:
            resnet.add(ResidualUnit(filters, strides=strides, kernel_size=kernel_size))
        prev_filters = filters

    return resnet

def get_resnet2(filters_sequence = [[64, 3, 5], [128, 4, 5], [256, 6, 5], [512, 3, 5]], input_shape=[400, 400, 1], first_kernel_size=7, se_blocks=False, activation='relu'):

    resnet = tf.keras.Sequential([
        DefaultConv2D(64, kernel_size=first_kernel_size, strides=2, input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=1, padding="same"),
    ])

    filters_list = [e for sublist in [[(f, kernel_size)]*n for f, n, kernel_size in filters_sequence] for e in sublist]
    prev_filters = filters_list[0]

    for filters, kernel_size in filters_list:
        strides = 1 if filters == prev_filters else 2
        if se_blocks:
            resnet.add(SE_ResidualUnit(filters, strides=strides, activation=activation))
        else:
            resnet.add(ResidualUnit(filters, strides=strides, kernel_size=kernel_size))
        prev_filters = filters

    return resnet

def get_SENet( num_blocks=4, filters=128, se_ratio=16, input_shape=[400, 400, 1], num_classes=2):
    model = Sequential()

    # Couche d'entrée
    model.add(DefaultConv2D(filters, kernel_size=7, activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))

    # Ajouter les blocs résiduels avec SE
    for _ in range(num_blocks):
        model.add(SE_ResidualUnit(filters, se_ratio=se_ratio))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2))



    return model


# Camnet
@keras.saving.register_keras_serializable()
class CAMNetS(tf.keras.Model):

    def __init__(self):
        super(CAMNetS, self).__init__()

        self.conv1 = Conv2D(32, 11, activation='relu', padding='same', dilation_rate=dilation_rate)
        self.conv2 = Conv2D(64, 7, activation='relu', padding='same', dilation_rate=dilation_rate)
        self.conv3 = Conv2D(128, 5, activation='relu', padding='same', dilation_rate=dilation_rate)
        self.conv4 = Conv2D(128, 3, activation='relu', padding='same', dilation_rate=dilation_rate)
        self.conv5 = Conv2D(64, 3, activation='relu', padding='same', dilation_rate=dilation_rate)

        self.mp = MaxPooling2D(2)
        
    def call(self, x):

        x = self.conv1(x)
        x = self.mp(x)
        x = self.conv2(x)
        x = self.mp(x)
        x = self.conv3(x)  
        x = self.conv4(x)
        x = self.conv5(x)

        return x

def get_CAMNetS(dilation_rate=1, **kwargs):
    return Sequential([
                    Conv2D(32, 11, activation='relu', padding='same', dilation_rate=dilation_rate),
                    MaxPooling2D(2),
                    Conv2D(64, 7, activation='relu', padding='same', dilation_rate=dilation_rate),
                    MaxPooling2D(2),
                    Conv2D(128, 5, activation='relu', padding='same', dilation_rate=dilation_rate),
                    Conv2D(128, 3, activation='relu', padding='same', dilation_rate=dilation_rate),
                    Conv2D(64, 3, activation='relu', padding='same', dilation_rate=dilation_rate)
        
    ])

def get_CAMNetS2(dilation_rate=1, **kwargs):
    return Sequential([
                    Conv2D(32, 11, activation='relu', padding='same', dilation_rate=dilation_rate),
                    MaxPooling2D(2),
                    Conv2D(64, 7, activation='relu', padding='same', dilation_rate=dilation_rate),
                    MaxPooling2D(2),
                    Conv2D(128, 5, activation='relu', padding='same', dilation_rate=dilation_rate),
                    MaxPooling2D(2),
                    Conv2D(128, 3, activation='relu', padding='same', dilation_rate=dilation_rate),
                    Conv2D(64, 3, activation='relu', padding='same', dilation_rate=dilation_rate)
        
    ])

class Hide_and_Seek(tf.keras.layers.Layer):
    def __init__(self, filters=(10, 10, 1), drop_rate=.5, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.drop_rate = drop_rate

    def call(self, x):
        grid = np.random.choice(a=[1, 0], p=[1-self.drop_rate, self.drop_rate], size=self.filters)
        grid = tf.where(tf.image.resize(grid, x.shape[1:3], method='area')[:, :, 0] > .5, 1.0, 0.0)
        grid = tf.expand_dims(grid, axis=-1)
        prod = x*grid
        return prod
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "drop_rate": self.drop_rate,
        })
        return config
    