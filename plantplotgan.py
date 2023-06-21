import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8 * 8 * 512, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 512)))

    model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(5, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model



# Critic Model
def build_critic():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[128, 128, 5]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# Define the loss function for the critic
def critic_loss(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

# Define the loss function for the generator
def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)

# Gradient penalty function
def spectral_information_divergence(critic, real_images, fake_images):
    BATCH_SIZE = tf.shape(real_images)[0]
    alpha = tf.random.uniform(shape=[BATCH_SIZE, 1, 1, 1], minval=0.0, maxval=1.0)
    real_images = tf.cast(real_images, tf.float32)
    interpolated_images = real_images[:BATCH_SIZE] + alpha * (fake_images[:BATCH_SIZE] - real_images[:BATCH_SIZE])

    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        critic_interpolated = critic(interpolated_images)

    sid = sid_loss(real_images, fake_images)
    return sid

def sid_loss(y_true, y_pred):
    epsilon = 1e-8
    y_true_normalized = y_true / (tf.reduce_sum(y_true, axis=[1, 2], keepdims=True) + epsilon)
    y_pred_normalized = y_pred / (tf.reduce_sum(y_pred, axis=[1, 2], keepdims=True) + epsilon)
    y_true_normalized_clipped = tf.clip_by_value(y_true_normalized, epsilon, 1.0)
    y_pred_normalized_clipped = tf.clip_by_value(y_pred_normalized, epsilon, 1.0)

    batch_size = tf.minimum(tf.shape(y_true_normalized_clipped)[0], tf.shape(y_pred_normalized_clipped)[0])
    y_true_normalized_clipped = y_true_normalized_clipped[:batch_size]
    y_pred_normalized_clipped = y_pred_normalized_clipped[:batch_size]
    sid = tf.reduce_sum(y_true_normalized_clipped * tf.math.log(y_true_normalized_clipped / y_pred_normalized_clipped), axis=[1, 2])

    return sid


def gradient_penalty(critic, real_images, fake_images):
    BATCH_SIZE = tf.shape(real_images)[0]
    alpha = tf.random.uniform(shape=[BATCH_SIZE, 1, 1, 1], minval=0.0, maxval=1.0)
    real_images = tf.cast(real_images, tf.float32)
    interpolated_images = real_images[:BATCH_SIZE] + alpha * (fake_images[:BATCH_SIZE] - real_images[:BATCH_SIZE])

    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        critic_interpolated = critic(interpolated_images)

    gradients = tape.gradient(critic_interpolated, interpolated_images)
    gradients_norm = tf.norm(gradients)
    gradient_penalty = tf.reduce_mean((gradients_norm - 1.0) ** 2)
    return gradient_penalty


# Training loop
def train_ppgan(images, epochs=100, batch_size=64, critic_steps=5):
    # Normalize the input images to the range [-1, 1]
    images = (images - 0.5) * 2.0

    generator = build_generator()
    critic = build_critic()

    # Define the optimizers for the generator and critic
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

    @tf.function
    def train_step(images):
        for i in range(critic_steps):
            # Generate random noise as input to the generator
            noise = tf.random.normal([batch_size, 100])
            print('critic range', i)
            with tf.GradientTape() as critic_tape:
                # Generate fake images from the noise using the generator
                generated_images = generator(noise)

                # Get the critic's output for real and fake images
                real_output = critic(images)
                fake_output = critic(generated_images)

                # Compute the critic loss and the gradient penalty
                critic_loss_value = critic_loss(real_output, fake_output)
                gp = gradient_penalty(critic, images, generated_images)
                total_loss = critic_loss_value + 10 * gp

            # Compute the gradients and update the critic's parameters
            critic_gradients = critic_tape.gradient(total_loss, critic.trainable_variables)
            critic_optimizer.apply_gradients(zip(critic_gradients, critic.trainable_variables))

        # Generate random noise as input to the generator
        noise = tf.random.normal([batch_size, 100])

        with tf.GradientTape() as generator_tape:
            # Generate fake images from the noise using the generator
            generated_images = generator(noise)

            # Get the critic's output for the generated images
            fake_output = critic(generated_images)

            # Compute the generator loss
            generator_loss_value = generator_loss(fake_output)

        # Compute the gradients and update the generator's parameters
        generator_gradients = generator_tape.gradient(generator_loss_value, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

        return critic_loss_value, generator_loss_value

    # Create a dataset from the input images
    dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(len(images)).batch(batch_size)

    # Training loop
    generator_wgan_loss_values = []
    for epoch in range(epochs):
        print(epoch)
        for image_batch in dataset:
            critic_loss_value, generator_loss_value = train_step(image_batch)

        generator_wgan_loss_values.append(generator_loss_value)
        # Print the losses for monitoring the training progress
        print(f"Epoch {epoch+1}/{epochs}, Critic Loss: {critic_loss_value:.4f}, Generator Loss: {generator_loss_value:.4f}")

    return generator, generator_wgan_loss_values

def train(EPOCHS, STEPS, BATCH_SIZE, NOISE_DIM, X_train_array):
    trained_generator, generator_loss = train_ppgan(X_train_array, EPOCHS, BATCH_SIZE, STEPS)
    return trained_generator, generator_loss