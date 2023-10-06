from keras.layers import MaxPooling2D, Conv2D, BatchNormalization, Dense, Dropout, Flatten, Input
from keras.models import Model, Sequential
from keras.optimizers import adam 
from keras import layers, Input
from keras.layers import Embedding

def my_model(input_shape, dropout_rt):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape = input_shape, activation = 'relu', data_format='channels_last'))
    model.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(dropout_rt))
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(dropout_rt))
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(dropout_rt)) 
    model.add(Dense(128, activation = 'relu'))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(dropout_rt))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model

def VGG3(input_shape, dropout_rt):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout_rt))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout_rt))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout_rt))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(dropout_rt))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def VGG2(input_shape, dropout_rt):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout_rt))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dropout(dropout_rt))
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(dropout_rt))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = adam(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model




# ResNet Model
def resnet50(input_shape, dropout_rt):
    inputs = Input(shape=input_shape)
    
    # Basic Residual Block
    def basic_block(x, filters, stride=1):
        # First convolution
        out = layers.Conv2D(filters, kernel_size=(3, 3), strides=stride, padding='same')(x)
        out = layers.BatchNormalization()(out)
        out = layers.ReLU()(out)
        out = layers.Dropout(dropout_rt)(out)
        

        # Second convolution
        out = layers.Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(out)
        out = layers.BatchNormalization()(out)

        # Shortcut connection
        if stride != 1:
            shortcut = layers.Conv2D(filters, kernel_size=(1, 1), strides=stride, padding='same')(x)
            shortcut = layers.BatchNormalization()(shortcut)
        else:
            shortcut = x

        out = layers.add([out, shortcut])
        out = layers.ReLU()(out)
        out = layers.Dropout(dropout_rt)(out)
        return out

    # Initial convolution
    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = layers.Dropout(dropout_rt)(x)

    # Residual blocks
    x = basic_block(x, 64, stride=1)
    x = basic_block(x, 64, stride=1)
    x = basic_block(x, 64, stride=1)

    x = basic_block(x, 128, stride=2)
    x = basic_block(x, 128, stride=1)
    x = basic_block(x, 128, stride=1)
    x = basic_block(x, 128, stride=1)

    x = basic_block(x, 256, stride=2)
    x = basic_block(x, 256, stride=1)
    x = basic_block(x, 256, stride=1)
    x = basic_block(x, 256, stride=1)
    x = basic_block(x, 256, stride=1)
    
    x = basic_block(x, 512, stride=2)
    x = basic_block(x, 512, stride=1)
    x = basic_block(x, 512, stride=1)

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected layer
    x = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs, x)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


import tensorflow as tf
# Define the multi-head self-attention layer
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = tf.reshape(
            query, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        key = tf.reshape(key, (batch_size, -1, self.num_heads, self.projection_dim))
        value = tf.reshape(
            value, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        query = tf.transpose(query, perm=[0, 2, 1, 3])
        key = tf.transpose(key, perm=[0, 2, 1, 3])
        value = tf.transpose(value, perm=[0, 2, 1, 3])

        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_scores = tf.multiply(
            attention_scores, tf.math.sqrt(tf.cast(self.projection_dim, tf.float32))
        )
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)
        output = tf.matmul(attention_scores, value)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(
            output, (batch_size, -1, self.embed_dim)
        )
        output = self.combine_heads(output)
        return output

# Define the feed-forward network layer
class FeedForwardNetwork(layers.Layer):
    def __init__(self, embed_dim, ff_dim):
        super(FeedForwardNetwork, self).__init__()
        self.dense1 = layers.Dense(ff_dim, activation="relu")
        self.dense2 = layers.Dense(embed_dim)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

# Define the Transformer Encoder layer
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = FeedForwardNetwork(embed_dim, ff_dim)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Define the Vision Transformer (ViT) model
def create_vit_model(input_shape, num_layers, embed_dim, num_heads, ff_dim, dropout_rt=0.1):
    inputs = Input(shape=input_shape)
    x = layers.Conv2D(embed_dim, 1, strides=1, padding="same")(inputs)
    x = layers.Reshape((-1, embed_dim))(x)
    num_patches = x.shape[1]
    patch_size = embed_dim // num_patches
    x = layers.Permute((2, 1))(x)

    # Positional embeddings
    positions = Embedding(input_dim=num_patches, output_dim=embed_dim)(tf.range(start=0, limit=num_patches, delta=1))
    x = x + positions

    for _ in range(num_layers):
        x = TransformerEncoder(embed_dim, num_heads, ff_dim, rate=dropout_rt)(x)

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rt)(x)
    x = layers.Dense(1, activation="sigmoid")(x)

    return Model(inputs, x)





