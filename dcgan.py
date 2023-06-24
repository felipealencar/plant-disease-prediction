from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, Reshape, Input, Conv2DTranspose
from keras.layers import Activation, LeakyReLU, BatchNormalization, Dropout, Resizing
import numpy as np 
from tqdm.auto import tqdm
from IPython.display import clear_output, display

def build_generator(optimizer, noise_dim, channels):
    model = Sequential([

        Dense(32*32*256, input_dim=noise_dim),
        LeakyReLU(alpha=0.2),
        Reshape((32,32,256)),
        
        Conv2DTranspose(128, (4, 4), strides=2, padding='same'),
        LeakyReLU(alpha=0.2),

        Conv2DTranspose(128, (4, 4), strides=2, padding='same'),
        LeakyReLU(alpha=0.2),

        Conv2D(channels, (4, 4), padding='same', activation='tanh')
    ], 
    name="generator")
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer=optimizer)

    return model


def build_discriminator(optimizer, width, height, channels):
    
    """
        Discriminator is the model which is responsible for classifying the generated images
        as fake or real. Our end goal is to create a Generator so powerful that the Discriminator
        is unable to classify real and fake images
        A simple Convolutional Neural Network with 2 Conv2D layers connected to a Dense output layer
        Output layer activation is Sigmoid since this is a Binary Classifier

        Input: Generated / Real Image
        Output: Validity of Image (Fake or Real)

    """

    model = Sequential([

        Conv2D(64, (3, 3), padding='same', input_shape=(width, height, channels)),
        LeakyReLU(alpha=0.2),

        Conv2D(128, (3, 3), strides=2, padding='same'),
        LeakyReLU(alpha=0.2),

        Conv2D(128, (3, 3), strides=2, padding='same'),
        LeakyReLU(alpha=0.2),
        
        Conv2D(256, (3, 3), strides=2, padding='same'),
        LeakyReLU(alpha=0.2),
        
        Flatten(),
        Dropout(0.4),
        Dense(1, activation="sigmoid", input_shape=(width, height, channels))
    ], name="discriminator")
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer=optimizer)

    return model


def build(optimizer, noise_dim, width, height, channels):
    discriminator = build_discriminator(optimizer, width, height, channels)
    generator = build_generator(optimizer, noise_dim, channels)
    trainable_discriminator_vars = discriminator.trainable_variables
    trainable_generator_vars = generator.trainable_variables
    trainable_vars = trainable_discriminator_vars + trainable_generator_vars
    optimizer.build(trainable_vars)
    #optimizer.apply_gradients(zip(trainable_vars, trainable_vars))
    discriminator.trainable = False 

    gan_input = Input(shape=(noise_dim,))
    fake_image = generator(gan_input)
    output = discriminator(fake_image)

    dcgan = Model(gan_input, output, name="gan_model")
    dcgan.compile(loss="binary_crossentropy", optimizer=optimizer)
    
    return generator, discriminator, dcgan


def train(X_train, generator, disciminator, model, noise, epochs, steps, batch_size, noise_dim):
    generator_dcgan_loss_values = []
    np.random.seed(40)
    for epoch in tqdm(range(epochs)):
        for _ in tqdm(range(steps)):

            noise = np.random.normal(0, 1, size=(batch_size, noise_dim))
            fake_X = generator.predict(noise)
            
            idx = np.random.randint(0, X_train.shape[0], size=batch_size)
            real_X = X_train[idx]
            
            X = np.concatenate((real_X, fake_X))
            disc_y = np.zeros(2*batch_size)
            
            disc_y[:batch_size] = 1
            d_loss = disciminator.train_on_batch(X, disc_y)
            
            y_gen = np.ones(batch_size)
            g_loss = model.train_on_batch(noise, y_gen)
            
        generator_dcgan_loss_values.append(g_loss)
        # Update the console output within the tqdm loop
        description = f"EPOCH: {epoch + 1} Generator Loss: {g_loss:.4f} Discriminator Loss: {d_loss:.4f}"
        clear_output(wait=True)
        display(description)
        noise = np.random.normal(0, 1, size=(batch_size, noise_dim))
    return generator, generator_dcgan_loss_values

