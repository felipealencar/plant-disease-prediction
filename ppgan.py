from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, Reshape, Input, Conv2DTranspose
from keras.layers import Activation, LeakyReLU, BatchNormalization, Dropout, Resizing
import numpy as np
import tensorflow as tf 
from tqdm import tqdm
from scipy.optimize import minimize
from tqdm.auto import tqdm
from IPython.display import clear_output, display

RED_EDGE_BAND = 3  # Index of the Red Edge band
NIR_BAND = 4  # Index of the Near Infrared band


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

    discriminator.trainable = False 

    gan_input = Input(shape=(noise_dim,))
    fake_image = generator(gan_input)
    output = discriminator(fake_image)

    gan = Model(gan_input, output, name="gan_model")
    gan.compile(loss="binary_crossentropy", optimizer=optimizer)
    return generator, discriminator, gan


def train(X, generator, disciminator, model, noise, epochs, steps, batch_size, noise_dim):
    generator_loss_values = []
    G, H, K = get_physics_covariance(X)
    for epoch in tqdm(range(epochs)):
        for _ in tqdm(range(steps)):
            noise = np.random.normal(0,1, size=(batch_size, noise_dim))
            fake_X = generator.predict(noise)

            # Extract the Red Edge and Near Infrared bands from the generated images
            red_edge = fake_X[:, :, 4]
            nir = fake_X[:, :, 5]

            # Apply the covariance constraint
            modified_red_edge = G * np.exp(-H * red_edge) + K * nir
            fake_X[:, :, RED_EDGE_BAND] = modified_red_edge

            idx = np.random.randint(0, X.shape[0], size=batch_size)
            real_X = X[idx]
            X = np.concatenate((real_X, fake_X))
            disc_y = np.zeros(2*batch_size)
            disc_y[:batch_size] = 1
            d_loss = disciminator.train_on_batch(X, disc_y)
            y_gen = np.ones(batch_size)
            g_loss = model.train_on_batch(noise, y_gen)
        generator_loss_values.append(g_loss)
        # Update the console output within the tqdm loop
        description = f"EPOCH: {epoch + 1} Generator Loss: {g_loss:.4f} Discriminator Loss: {d_loss:.4f}"
        clear_output(wait=True)
        display(description)
        noise = np.random.normal(0, 1, size=(batch_size, noise_dim))
    return generator, generator_loss_values


def red_edge_loss(coefficients, red_edge, nir, desired_covariance):
    """
    Calculates the discrepancy between the covariance of modified Red Edge band and the desired covariance.

    Args:
        coefficients (list): A list of coefficients [G, H, K].
        red_edge (numpy.ndarray): Red Edge band data.
        nir (numpy.ndarray): Near Infrared band data.
        desired_covariance (float): The desired covariance value.

    Returns:
        float: The discrepancy between the calculated covariance and the desired covariance.
    """
    G, H, K = coefficients
    modified_red_edge = G * np.exp(H * red_edge) + K * nir
    covariance = np.cov(modified_red_edge.flatten(), nir.flatten())[0, 1]
    discrepancy = abs(covariance - desired_covariance)
    return discrepancy


def get_physics_covariance(X):
    """
    Calculates the optimized coefficients for the PlantPlotGAN model based on covariance analysis.

    Args:
        X (numpy.ndarray): Input data array containing multiple bands.

    Returns:
        tuple: A tuple containing the optimized coefficients (G_optimized, H_optimized, K_optimized).
    """

    # Extract the Red Edge and Near Infrared bands from X
    red_edge = X[:, :, :, RED_EDGE_BAND]
    nir = X[:, :, :, NIR_BAND]

    # Reshape the bands for covariance calculation
    red_edge_reshaped = np.reshape(red_edge, (20, -1))
    nir_reshaped = np.reshape(nir, (20, -1))

    # Compute the covariance matrix
    covariance_matrix = np.cov(red_edge_reshaped, nir_reshaped)

    # Extract the covariance between the Red Edge and Near Infrared bands
    covariance = covariance_matrix[0, 1]

    desired_covariance = covariance

    # Define the initial guess for the coefficients
    initial_guess = [1.0, 1.0, 1.0]

    # Define the bounds for the coefficients if necessary
    bounds = [(0, None), (None, None), (0, None)]

    # Perform optimization
    result = minimize(red_edge_loss, initial_guess, args=(red_edge, nir, desired_covariance), bounds=bounds)

    # Get the optimized coefficients
    optimized_coefficients = result.x
    print('Obtained coefficients:', optimized_coefficients)
    G_optimized, H_optimized, K_optimized = optimized_coefficients

    return G_optimized, H_optimized, K_optimized


def sid_loss(y_true, y_pred):
    epsilon = 1e-8
    print('y_pred', y_pred)
    y_true_normalized = y_true / (tf.reduce_sum(y_true, axis=[1], keepdims=True) + epsilon)
    y_pred_normalized = y_pred / (tf.reduce_sum(y_pred, axis=[1], keepdims=True) + epsilon)
    y_true_normalized_clipped = tf.clip_by_value(y_true_normalized, epsilon, 1.0)
    y_pred_normalized_clipped = tf.clip_by_value(y_pred_normalized, epsilon, 1.0)
    print('y_pred_norm_clipped', y_pred_normalized_clipped)
    batch_size = tf.minimum(tf.shape(y_true_normalized_clipped)[0], tf.shape(y_pred_normalized_clipped)[0])
    y_true_normalized_clipped = y_true_normalized_clipped[:batch_size]
    y_pred_normalized_clipped = y_pred_normalized_clipped[:batch_size]
    sid = tf.reduce_sum(y_true_normalized_clipped * tf.math.log(y_true_normalized_clipped / y_pred_normalized_clipped), axis=[1])
    
    return tf.reduce_mean(sid)