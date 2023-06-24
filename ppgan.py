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
    #model.summary()
    #model.compile(loss="binary_crossentropy", optimizer=optimizer)

    return model


def build_discriminator(optimizer, width, height, channels, name='discriminator', loss='binary_crossentropy'):
    
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
    ], name=name)
    #model.summary()
    #model.compile(loss=loss, optimizer=optimizer)

    return model



def build(optimizer, noise_dim, width, height, channels):
    discriminator_crossentropy = build_discriminator(optimizer, width, height, channels, 'd1')
    discriminator_sid = build_discriminator(optimizer, width, height, channels, 'd2', sid_loss)
    generator = build_generator(optimizer, noise_dim, channels)
    trainable_discriminator_cross_vars = discriminator_crossentropy.trainable_variables
    trainable_discriminator_sid_vars = discriminator_sid.trainable_variables
    trainable_generator_vars = generator.trainable_variables
    trainable_vars = trainable_discriminator_cross_vars + trainable_discriminator_sid_vars + trainable_generator_vars
    optimizer.build(trainable_vars)

    discriminator_crossentropy.trainable = False 
    discriminator_sid.trainable = False 

    gan_input = Input(shape=(noise_dim,))
    fake_image = generator(gan_input)
    output_1 = discriminator_crossentropy(fake_image)
    output_2 = discriminator_sid(fake_image)
    output = output_1 * output_2
    gan = Model(gan_input, output, name="gan_model")
    gan.compile(loss="binary_crossentropy", optimizer=optimizer)
    return generator, discriminator_crossentropy, discriminator_sid, gan


@tf.function
def train_step(real_data, generator, discriminator1, discriminator2, optimizer_G, optimizer_D1, optimizer_D2, adversarial_loss, spectral_regularization, batch_size, noise_dim, G, H, K):
    noise = np.random.normal(0, 1, size=(batch_size, noise_dim))
    # fake_X = generator.predict(noise)

    # # Extract the Red Edge and Near Infrared bands from the generated images
    # red_edge = fake_X[:, :, 4]
    # nir = fake_X[:, :, 5]

    # # Apply the covariance constraint
    # modified_red_edge = G * np.exp(-H * red_edge) + K * nir
    # fake_X[:, :, RED_EDGE_BAND] = red_edge-modified_red_edge

    # idx = np.random.randint(0, real_data.shape[0], size=batch_size)
    # real_X = real_data[idx]
    # X = np.concatenate((real_X, fake_X))
    # disc_y = np.zeros(2*batch_size)
    # disc_y[:batch_size] = 1
    # d1_loss = discriminator1.train_on_batch(X, disc_y)
    # d2_loss = discriminator2.train_on_batch(X, disc_y)
    # y_gen = np.ones(batch_size)
    # g_loss = model.train_on_batch(noise, y_gen)
    # batch_size = tf.shape(real_data)[0]
    spectral_weight = 1.0
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc1_tape, tf.GradientTape() as disc2_tape:
        # Generate fake data
        fake_data = generator(noise, training=True)
        red_edge = fake_data[:, :, :, 3]
        nir = fake_data[:, :, :, 4]
        G = tf.cast(G, tf.float32)
        H = tf.cast(H, tf.float32)
        K = tf.cast(K, tf.float32)
        modified_red_edge = G * tf.exp(-H * red_edge) + K * nir
        # Update the last channel of `fake_data` with the modified Red Edge band
        # Replace the values of the last channel of `fake_data` with the values of `modified_red_edge`
        last_channel_index = tf.shape(fake_data)[-1] - 1
        fake_data2 = tf.concat([fake_data[:, :, :, :last_channel_index], modified_red_edge[..., tf.newaxis]], axis=-1)
        min_value = tf.reduce_min(fake_data2)
        fake_data2 = fake_data2 / min_value

        # Compute discriminator losses
        real_output1 = discriminator1(real_data, training=True)
        fake_output1 = discriminator1(fake_data, training=True)
        disc1_loss = adversarial_loss(tf.ones_like(real_output1), real_output1) * adversarial_loss(tf.zeros_like(fake_output1), fake_output1)

         # Calculate spectral regularization loss
        real_data = tf.cast(real_data, dtype=tf.float32)
        fake_data = tf.cast(fake_data, dtype=tf.float32)
        real_spectrum = tf.signal.fftshift(tf.signal.fft2d(tf.complex(real_data, 0.0)))
        fake_spectrum = tf.signal.fftshift(tf.signal.fft2d(tf.complex(fake_data, 0.0)))
        sr_loss = spectral_regularization(real_spectrum, fake_spectrum)
        print(sr_loss)
        
        real_output2 = discriminator2(real_spectrum, training=True)
        fake_output2 = discriminator2(fake_spectrum, training=True)
        disc2_loss = adversarial_loss(tf.ones_like(real_output2), real_output2) * adversarial_loss(tf.zeros_like(fake_output2), fake_output2)
        adversarial_loss(tf.ones_like(real_output2), real_output2) * adversarial_loss(tf.zeros_like(fake_output2), fake_output2)
        #disc2_loss = disc2_loss + (tf.cast(sr_loss, tf.float32) * tf.cast(spectral_weight, tf.float32))


        # Total generator loss
        gen_loss = adversarial_loss(tf.ones_like(fake_output1), fake_output1) + (adversarial_loss(tf.ones_like(fake_output2), fake_output2)) + (tf.cast(spectral_weight, tf.float32) * tf.cast(sr_loss, tf.float32))

    # Compute gradients
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator1 = disc1_tape.gradient(disc1_loss, discriminator1.trainable_variables)
    gradients_of_discriminator2 = disc2_tape.gradient(disc2_loss, discriminator2.trainable_variables)

    # Update weights
    optimizer_G.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    optimizer_D1.apply_gradients(zip(gradients_of_discriminator1, discriminator1.trainable_variables))
    optimizer_D2.apply_gradients(zip(gradients_of_discriminator2, discriminator2.trainable_variables))

    return gen_loss, disc1_loss, disc2_loss

def train(X_train, generator, discriminator1, discriminator2, model, epochs, steps, batch_size, noise_dim):
    # Define the loss functions
    adversarial_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    spectral_regularization = tf.keras.losses.MeanSquaredError()
    # Define the optimizer for the generator and discriminator
    optimizer_G = tf.keras.optimizers.legacy.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
    optimizer_D1 = tf.keras.optimizers.legacy.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
    optimizer_D2 = tf.keras.optimizers.legacy.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)

    generator_loss_values = []
    np.random.seed(40)
    G, H, K = get_physics_covariance(X_train)
    for epoch in tqdm(range(epochs)):
        for _ in tqdm(range(steps)):
            gen_loss, disc1_loss, disc2_loss = train_step(X_train, generator, discriminator1, discriminator2, optimizer_G, optimizer_D1, optimizer_D2, adversarial_loss, spectral_regularization, batch_size, noise_dim, G, H, K)

        generator_loss_values.append(gen_loss)
        # Update the console output within the tqdm loop
        description = f"EPOCH: {epoch + 1} Generator Loss: {gen_loss:.4f} Discriminator 1 Loss: {disc1_loss:.4f} Discriminator 2 Loss: {disc2_loss:.4f}"
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
    print('y_true', y_true)
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