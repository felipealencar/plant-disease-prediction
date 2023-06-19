from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, Reshape, Input, Conv2DTranspose
from keras.layers import Activation, LeakyReLU, BatchNormalization, Dropout, Resizing
import numpy as np 
from tqdm import tqdm


def build_generator(OPTIMIZER, NOISE_DIM):
    model = Sequential([
        Dense(25 * 25 * 256, input_dim=NOISE_DIM),
        LeakyReLU(alpha=0.2),
        Reshape((25, 25, 256)),
        
        Conv2DTranspose(32, (2, 2), strides=(3, 3), padding='valid'),
        LeakyReLU(alpha=0.2),

        Conv2DTranspose(64, (2, 2), strides=(1, 1), padding='valid'),
        LeakyReLU(alpha=0.2),
        
        Conv2DTranspose(128, (2, 2), strides=(1, 1), padding='valid'),
        LeakyReLU(alpha=0.2),

        Conv2D(2, (3, 3), padding='valid', activation='tanh')
    ], name="generator")

    model.summary()
    model.compile(loss="binary_crossentropy", optimizer=OPTIMIZER)

    return model


def build_discriminator(OPTIMIZER):
    model = Sequential([
        Conv2D(64, (3, 3), padding='same', input_shape=(75, 75, 2)),
        LeakyReLU(alpha=0.2),

        Conv2D(75, (3, 3), strides=2, padding='same'),
        LeakyReLU(alpha=0.2),

        Conv2D(75, (3, 3), strides=2, padding='same'),
        LeakyReLU(alpha=0.2),

        Conv2D(75, (3, 3), strides=2, padding='same'),
        LeakyReLU(alpha=0.2),

        Flatten(),
        Dropout(0.4),
        Dense(1, activation="sigmoid")
    ], name="discriminator")
    
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer=OPTIMIZER)
    
    return model


def build(OPTIMIZER, NOISE_DIM):
    print('\n')
    discriminator = build_discriminator(OPTIMIZER)
    print('\n')
    generator = build_generator(OPTIMIZER, NOISE_DIM)
    trainable_discriminator_vars = discriminator.trainable_variables
    trainable_generator_vars = generator.trainable_variables
    trainable_vars = trainable_discriminator_vars + trainable_generator_vars
    OPTIMIZER.build(trainable_vars)

    discriminator.trainable = False 

    gan_input = Input(shape=(NOISE_DIM,))
    fake_image = generator(gan_input)
    output = discriminator(fake_image)

    dcgan = Model(gan_input, output, name="gan_model")
    dcgan.compile(loss="binary_crossentropy", optimizer=OPTIMIZER)
    return generator, discriminator, dcgan


def train(generator, disciminator, model, noise, EPOCHS, STEPS, BATCH_SIZE, NOISE_DIM, X_train_array):
    generator_dcgan_loss_values = []
    for epoch in range(EPOCHS):
        for _ in tqdm(range(STEPS)):
            noise = np.random.normal(0,1, size=(BATCH_SIZE, NOISE_DIM))
            fake_X = generator.predict(noise)
            idx = np.random.randint(0, X_train_array.shape[0], size=BATCH_SIZE)
            real_X = X_train_array[idx]
            X = np.concatenate((real_X, fake_X))
            disc_y = np.zeros(2*BATCH_SIZE)
            disc_y[:BATCH_SIZE] = 1
            d_loss = disciminator.train_on_batch(X, disc_y)
            y_gen = np.ones(BATCH_SIZE)
            g_loss = model.train_on_batch(noise, y_gen)
        generator_dcgan_loss_values.append(g_loss)
        print(f"EPOCH: {epoch + 1} Generator Loss: {g_loss:.4f} Discriminator Loss: {d_loss:.4f}")
        noise = np.random.normal(0, 1, size=(BATCH_SIZE, NOISE_DIM))
    return generator_dcgan_loss_values