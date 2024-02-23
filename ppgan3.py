from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, Reshape, Input, Conv2DTranspose
from keras.layers import Activation, LeakyReLU, BatchNormalization, Dropout
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tqdm import tqdm
from scipy.optimize import minimize
from tqdm.auto import tqdm
from IPython.display import clear_output, display
import sys

sys.path.append('../temporal-multispectral-gen-models')

import dgl
import gnn as gnn
import importlib 
importlib.reload(gnn)
from gnn.gnn import BaseGNNModel


import utils 
importlib.reload(utils)
from utils import graph_segmentation
from gnn.utils import discriminate_generated_graphs

import torch
import torch.nn.functional as F

RED_EDGE_BAND = 3  # Index of the Red Edge band
NIR_BAND = 4  # Index of the Near Infrared band

def build_generator(optimizer, noise_dim, channels, name="generator"):
    model = Sequential(
        [
            Dense(32 * 32 * 256, input_dim=noise_dim),
            LeakyReLU(alpha=0.2),
            Reshape((32, 32, 256)),
            Conv2DTranspose(128, (4, 4), strides=2, padding="same"),
            LeakyReLU(alpha=0.2),
            Conv2DTranspose(128, (4, 4), strides=2, padding="same"),
            LeakyReLU(alpha=0.2),
            Conv2D(channels, (4, 4), padding="same", activation="tanh"),
        ],
        name=name,
    )
    # model.summary()
    model.compile(loss="binary_crossentropy", optimizer=optimizer)

    return model


def build_discriminator(
    optimizer, width, height, channels, name="discriminator", loss="binary_crossentropy"
):
    """
    Discriminator is the model which is responsible for classifying the generated images
    as fake or real. Our end goal is to create a Generator so powerful that the Discriminator
    is unable to classify real and fake images
    A simple Convolutional Neural Network with 2 Conv2D layers connected to a Dense output layer
    Output layer activation is Sigmoid since this is a Binary Classifier

    Input: Generated / Real Image
    Output: Validity of Image (Fake or Real)

    """

    model = Sequential(
        [
            Conv2D(64, (3, 3), padding="same", input_shape=(width, height, channels)),
            LeakyReLU(alpha=0.2),
            Conv2D(128, (3, 3), strides=2, padding="same"),
            LeakyReLU(alpha=0.2),
            Conv2D(128, (3, 3), strides=2, padding="same"),
            LeakyReLU(alpha=0.2),
            Conv2D(256, (3, 3), strides=2, padding="same"),
            LeakyReLU(alpha=0.2),
            Flatten(),
            Dropout(0.4),
            Dense(1, activation="sigmoid", input_shape=(width, height, channels)),
        ],
        name=name,
    )
    # model.summary()
    model.compile(loss=loss, optimizer=optimizer)

    return model


def build(optimizer, noise_dim, width, height, channels):
    discriminator_crossentropy = build_discriminator(
        optimizer, width, height, channels, "d1"
    )
    discriminator_sid = build_discriminator(
        optimizer, width, height, channels, "d2", sid_loss
    )
    discriminator_gnn = build_discriminator(
        optimizer, width, height, channels, "d3", gnn_loss
    )
    generator1 = build_generator(optimizer, noise_dim, channels, "g1")
    generator2 = build_generator(optimizer, noise_dim, channels, "g2")
    trainable_discriminator_cross_vars = discriminator_crossentropy.trainable_variables
    trainable_discriminator_sid_vars = discriminator_sid.trainable_variables
    trainable_discriminator_gnn_vars = discriminator_gnn.trainable_variables
    trainable_generator1_vars = generator1.trainable_variables
    trainable_generator2_vars = generator2.trainable_variables
    trainable_vars = (
        trainable_discriminator_cross_vars
        + trainable_discriminator_sid_vars
        + trainable_discriminator_gnn_vars
        + trainable_generator1_vars
        + trainable_generator2_vars
    )
    optimizer.build(trainable_vars)

    discriminator_crossentropy.trainable = False
    discriminator_sid.trainable = False
    discriminator_gnn.trainable = False

    gan_input = Input(shape=(noise_dim,))
    fake_image1 = generator1(gan_input)
    fake_image2 = generator2(gan_input)
    output_1 = discriminator_crossentropy(fake_image1)
    output_2 = discriminator_sid(fake_image1)
    output_3 = discriminator_gnn(fake_image2)
    output = output_1 * output_2 * output_3
    gan = Model(gan_input, output, name="gan_model")
    gan.compile(loss="binary_crossentropy", optimizer=optimizer, run_eagerly=True)
    return generator1, generator2, discriminator_crossentropy, discriminator_sid, discriminator_gnn, gan


#@tf.function
def train_step(
    real_data,
    generator1,
    generator2,
    gnn,
    discriminator1,
    discriminator2,
    discriminator3,
    optimizer_G1,
    optimizer_G2,
    optimizer_D1,
    optimizer_D2,
    optimizer_D3,
    adversarial_loss,
    spectral_regularization,
    gnn_loss,
    batch_size,
    noise_dim,
    G,
    H,
    K,
):
    noise = np.random.normal(0, 1, size=(batch_size, noise_dim))
    spectral_weight = 5.0
    temporal_weight = 5.0
    np.random.seed(42)
    idxs = np.random.randint(0, real_data.shape[0], batch_size)
    
    real_data = real_data[idxs]
    with tf.GradientTape() as gen1_tape, tf.GradientTape() as gen2_tape, tf.GradientTape() as disc1_tape, tf.GradientTape() as disc2_tape, tf.GradientTape() as disc3_tape:
        imagery = real_data
        
        # Generate fake data
        fake_data = generator1(noise, training=True)
        fake_data2 = generator2(noise, training=True)
        red_edge = fake_data[:, :, :, 3]
        nir = fake_data[:, :, :, 4]
        G = tf.cast(G, tf.float32)
        H = tf.cast(H, tf.float32)
        K = tf.cast(K, tf.float32)
        modified_red_edge = G * tf.exp(-H * red_edge) + K * nir
        # Update the last channel of `fake_data` with the modified Red Edge band
        # Replace the values of the last channel of `fake_data` with the values of `modified_red_edge`
        last_channel_index = tf.shape(fake_data)[-1] - 1
        fake_data_regularized = tf.concat(
            [
                fake_data[:, :, :, :last_channel_index],
                modified_red_edge[..., tf.newaxis],
            ],
            -1,
        )
        min_value = tf.reduce_min(fake_data_regularized)
        fake_data_regularized = fake_data_regularized / min_value

        min_value = tf.reduce_min(fake_data2)
        fake_data2 = fake_data2 / min_value

        # Compute discriminator losses
        real_output1 = discriminator1(real_data, training=True)
        fake_output1 = discriminator1(fake_data, training=True)
        disc1_loss = adversarial_loss(
            tf.ones_like(real_output1), real_output1
        ) * adversarial_loss(tf.zeros_like(fake_output1), fake_output1)

        # Calculate spectral regularization loss
        real_data = tf.cast(real_data, dtype=tf.float32)
        fake_data_regularized = tf.cast(fake_data_regularized, dtype=tf.float32)        
        real_spectrum = tf.signal.fftshift(tf.signal.fft2d(tf.complex(real_data, 0.0)))
        fake_spectrum = tf.signal.fftshift(tf.signal.fft2d(tf.complex(fake_data_regularized, 0.0)))
        sr_loss = spectral_regularization(real_spectrum, fake_spectrum)
        
        # Adicionei o spectrum como parâmetro do discriminator 2
        real_output2 = discriminator2(real_spectrum, training=True)
        fake_output2 = discriminator2(fake_spectrum, training=True)
        disc2_loss = (
            adversarial_loss(tf.ones_like(real_output2), real_output2)
            * adversarial_loss(tf.zeros_like(fake_output2), fake_output2)
        )

        tf.config.run_functions_eagerly(True)  # Enable eager execution
        
        fake_array = fake_data2.numpy()

        tf.config.run_functions_eagerly(False) 
        
        # TODO: remember to parameterize the graph_segmentation function for num_days

        fake_graph = graph_segmentation(fake_array[0], 50, 100)
        real_graph = graph_segmentation(imagery[0], 50, 100)
        
        fake_graph.ndata['feat'][torch.isnan(fake_graph.ndata['feat'])] = 0
        real_graph.ndata['feat'][torch.isnan(real_graph.ndata['feat'])] = 0

        # Make predictions
        fake_graph = dgl.add_self_loop(fake_graph)
        real_graph = dgl.add_self_loop(real_graph)
        
        output = discriminate_generated_graphs(gnn, fake_graph)


        scaler = MinMaxScaler()
        real_graph_normalized_features = scaler.fit_transform(real_graph.ndata['feat'])
        output_normalized_features = scaler.fit_transform(output)
        g_loss = gnn_loss(real_graph_normalized_features, output_normalized_features)
        print('g_loss', g_loss)
        # Removi a constante 10
        # Calculate spectral regularization loss
        real_data = tf.cast(real_data, dtype=tf.float32)
        fake_data = tf.cast(fake_data, dtype=tf.float32)
        real_spectrum = tf.signal.fftshift(tf.signal.fft2d(tf.complex(real_data, 0.0)))
        fake_spectrum = tf.signal.fftshift(tf.signal.fft2d(tf.complex(fake_data, 0.0)))
        sr_loss = spectral_regularization(real_spectrum, fake_spectrum)

        real_output3 = discriminator3(real_data, training=True)
        fake_output3 = discriminator3(fake_data2, training=True)
        disc3_loss = (
            adversarial_loss(tf.ones_like(real_output3), real_output3)
            * adversarial_loss(tf.zeros_like(fake_output3), fake_output3)
        )

        # Adicionei uma nova adversarial loss e removi o spectral_weight
        # Total generator loss
        gen1_loss = (
            adversarial_loss(tf.ones_like(fake_output1), fake_output1)
            + (adversarial_loss(tf.ones_like(fake_output2), fake_output2))
            + (adversarial_loss(tf.zeros_like(real_output1), real_output1))
            + (adversarial_loss(tf.zeros_like(real_output2), real_output2))
            + (tf.cast(sr_loss, tf.float32) * spectral_weight)
        )
        gen2_loss = (
            adversarial_loss(tf.ones_like(fake_output3), fake_output3)
            + (adversarial_loss(tf.zeros_like(real_output3), real_output3))
            + (tf.cast(g_loss, tf.float32))
        )
        print('gen1_loss', gen1_loss)
        print('gen2_loss', gen2_loss)

    print('teste')
    # Compute gradients
    gradients_of_generator1 = gen1_tape.gradient(gen1_loss, generator1.trainable_variables)
    gradients_of_generator2 = gen2_tape.gradient(gen2_loss, generator2.trainable_variables)

    gradients_of_discriminator1 = disc1_tape.gradient(
        disc1_loss, discriminator1.trainable_variables
    )
    gradients_of_discriminator2 = disc2_tape.gradient(
        disc2_loss, discriminator2.trainable_variables
    )
    gradients_of_discriminator3 = disc3_tape.gradient(
        disc3_loss, discriminator3.trainable_variables
    )

    # Update weights
    optimizer_G1.apply_gradients(
        zip(gradients_of_generator1, generator1.trainable_variables)
    )
    optimizer_G2.apply_gradients(
        zip(gradients_of_generator2, generator2.trainable_variables)
    )
    optimizer_D1.apply_gradients(
        zip(gradients_of_discriminator1, discriminator1.trainable_variables)
    )
    optimizer_D2.apply_gradients(
        zip(gradients_of_discriminator2, discriminator2.trainable_variables)
    )
    optimizer_D3.apply_gradients(
        zip(gradients_of_discriminator3, discriminator3.trainable_variables)
    )

    return gen1_loss, gen2_loss, disc1_loss, disc2_loss, disc3_loss


def train(
    X_train,
    generator1,
    generator2,
    discriminator1,
    discriminator2,
    discriminator3,
    model,
    epochs,
    steps,
    batch_size,
    noise_dim,
):
    # Define the loss functions
    adversarial_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    spectral_regularization = tf.keras.losses.MeanSquaredError()
    gnn_loss = tf.keras.losses.MeanSquaredError()

    # Define the optimizer for the generator and discriminator
    optimizer_G1 = tf.keras.optimizers.Adam(0.0002, 0.5)
    optimizer_G2 = tf.keras.optimizers.Adam(0.0002, 0.5)
    optimizer_D1 = tf.keras.optimizers.Adam(0.0002, 0.5)
    optimizer_D2 = tf.keras.optimizers.Adam(0.0002, 0.5)
    optimizer_D3 = tf.keras.optimizers.Adam(0.0002, 0.5)

    generator1_loss_values = []
    generator2_loss_values = []
    np.random.seed(40)
    G, H, K = get_physics_covariance(X_train)

    gnn = BaseGNNModel(input_dim=6, hidden_dim=256, output_dim=5, num_layers=12)
    # Load the state dictionary of the saved model
    gnn.load_state_dict(torch.load(r'C:\Users\flopes1\OneDrive - Saint Louis University\Desktop\Repos\temporal-multispectral-gen-models\gnn\trained_backcasting_gnn_model.pth'))
    # Set the model in evaluation mode
    gnn.eval()
    
    for epoch in tqdm(range(epochs)):
        for _ in tqdm(range(steps)):
            gen1_loss, gen2_loss, disc1_loss, disc2_loss, disc3_loss = train_step(
                X_train,
                generator1,
                generator2,
                gnn,
                discriminator1,
                discriminator2,
                discriminator3,
                optimizer_G1,
                optimizer_G2,
                optimizer_D1,
                optimizer_D2,
                optimizer_D3,
                adversarial_loss,
                spectral_regularization,
                gnn_loss,
                batch_size,
                noise_dim,
                G,
                H,
                K,
            )
            print(_)

        generator1_loss_values.append(gen1_loss)
        generator2_loss_values.append(gen2_loss)
        # Update the console output within the tqdm loop
        description = f"EPOCH: {epoch + 1} Generator 1 Loss: {gen1_loss:.4f} Generator 2 Loss: {gen2_loss:.4f} Discriminator 1 Loss: {disc1_loss:.4f} Discriminator 2 Loss: {disc2_loss:.4f} Discriminator 3 Loss: {disc3_loss:.4f}"
        clear_output(wait=True)
        display(description)
    return generator1, generator1_loss_values, generator2, generator2_loss_values


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
    red_edge_reshaped = np.reshape(red_edge, (X.shape[0], -1))
    nir_reshaped = np.reshape(nir, (X.shape[0], -1))

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
    result = minimize(
        red_edge_loss,
        initial_guess,
        args=(red_edge, nir, desired_covariance),
        bounds=bounds,
    )

    # Get the optimized coefficients
    optimized_coefficients = result.x
    print("Obtained coefficients:", optimized_coefficients)
    G_optimized, H_optimized, K_optimized = optimized_coefficients

    return G_optimized, H_optimized, K_optimized


def sid_loss(y_true, y_pred):
    epsilon = 1e-8
    y_true_normalized = y_true / (
        tf.reduce_sum(y_true, axis=[1], keepdims=True) + epsilon
    )
    y_pred_normalized = y_pred / (
        tf.reduce_sum(y_pred, axis=[1], keepdims=True) + epsilon
    )
    y_true_normalized_clipped = tf.clip_by_value(y_true_normalized, epsilon, 1.0)
    y_pred_normalized_clipped = tf.clip_by_value(y_pred_normalized, epsilon, 1.0)
    batch_size = tf.minimum(
        tf.shape(y_true_normalized_clipped)[0], tf.shape(y_pred_normalized_clipped)[0]
    )
    y_true_normalized_clipped = y_true_normalized_clipped[:batch_size]
    y_pred_normalized_clipped = y_pred_normalized_clipped[:batch_size]
    sid = tf.reduce_sum(
        y_true_normalized_clipped
        * tf.math.log(y_true_normalized_clipped / y_pred_normalized_clipped),
        axis=[1],
    )

    return tf.reduce_mean(sid)

def tensor_to_array(tensor):
    return tensor.numpy()

def gnn_loss(y_true, y_pred):
    # Assuming y_true and y_pred are DGL graphs with node features 'feat'

    # Check if the number of nodes and edges are the same in both graphs
    assert y_true.number_of_nodes() == y_pred.number_of_nodes(), "Number of nodes mismatch"
    assert y_true.number_of_edges() == y_pred.number_of_edges(), "Number of edges mismatch"

    # Calculate the loss based on node features
    true_feats = y_true.ndata['feat']
    pred_feats = y_pred.ndata['feat']

    # Assuming a simple L2 loss between node features
    loss = F.mse_loss(true_feats, pred_feats)

    return loss


def convert_to_numpy(tensor):
    return tensor.numpy()

@tf.function
def process_data(data):
    numpy_data = tf.py_function(func=convert_to_numpy, inp=[data], Tout=tf.float32)
    return numpy_data