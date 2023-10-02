from keras.layers import MaxPooling2D, Conv2D, BatchNormalization, Dense, Dropout, Flatten, Input
from keras.models import Model, Sequential
from keras.optimizers import adam 

def my_model(filter_size, INPUT_SHAPE, dropout_rt):
    model = Sequential()
    model.add(Conv2D(filter_size, (3, 3), input_shape = INPUT_SHAPE, activation = 'relu', data_format='channels_last'))
    model.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(dropout_rt))
    model.add(Conv2D(filter_size, (3, 3), activation = 'relu'))
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

def VGG3(INPUT_SHAPE, dropout_rt):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=INPUT_SHAPE))
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


def VGG2(INPUT_SHAPE, dropout_rt):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=INPUT_SHAPE))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
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
