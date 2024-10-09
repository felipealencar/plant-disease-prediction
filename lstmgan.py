import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
from dotenv import load_dotenv

def build_generator(sequence_length, latent_dim, num_features):
    model = models.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(16 * 16 * num_features, activation='relu'),  
        layers.Reshape((16, 16, num_features)),  
        layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        layers.Conv2DTranspose(10, (3, 3), strides=(2, 2), padding='same', activation='sigmoid')
    ])
    return model

def build_discriminator(sequence_length, num_features, channels):
    model = models.Sequential([
        layers.Input(shape=(128, 128, channels)),  # Adjusted input shape
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.25),
        layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(sequence_length * 128),  # Adjusted dense layer
        layers.Reshape((sequence_length, 128)),  # Reshape to fit LSTM input
        layers.LSTM(128),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model




def find_next_different_day_index(idx, y_train):
    current_day = y_train[idx, -1]
    next_idx = idx + 1
    while next_idx < len(y_train) and y_train[next_idx, -1] == current_day:
        next_idx += 1
    return next_idx

def train(X_train, y_train, generator, discriminator, combined, epochs=100, batch_size=128, sample_interval=10, latent_dim=100):
    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs*10):
        # Train discriminator so it discriminates between a sample from instant t and a sample from instant t+1
        # The samples are images of 128x128x5x1 (with the number of days as the last column)
        # The discriminator should be able to tell if the sample is from the next day sequence or not
        # The generator should be able to generate a sequence that is similar to the real one
        print(X_train[:42].shape[0])
        idx = np.random.randint(0, X_train[:42].shape[0] - 1, batch_size)
        next_idx = [find_next_different_day_index(i, y_train) for i in idx]
        
        real_sequences_current = X_train[idx]
        real_sequences_next = X_train[next_idx]
        
        # Concatenate current and next sequences along the last axis
        real_sequences = np.concatenate([real_sequences_current, real_sequences_next], axis=-1)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_sequences = generator.predict(noise)
        print(real_sequences.shape)
        print(generated_sequences.shape)
        d_loss_real = discriminator.train_on_batch(real_sequences, valid)
        d_loss_fake = discriminator.train_on_batch(generated_sequences, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = combined.train_on_batch(noise, valid)

        # Print progress
        if epoch % sample_interval == 0:
            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}] [G loss: {g_loss}]")

    # Save the model
    generator.save('generator_lstmgan.h5')

# Example usage
# train(X_train)
