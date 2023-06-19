import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Generator Model
def build_generator_wgan():
    model = tf.keras.Sequential()
    model.add(layers.Dense(18 * 18 * 256, use_bias=False, input_shape=(75 * 75 * 2,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((18, 18, 256)))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(2, (4, 4), strides=(1, 1), padding='valid', use_bias=False, activation='tanh'))

    return model


# Critic Model
def build_critic_wgan():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[75, 75, 2]))
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
def train_wgan(images, epochs=100, batch_size=64, critic_steps=5):
    # Normalize the input images to the range [-1, 1]
    images = (images - 0.5) * 2.0

    generator = build_generator_wgan()
    critic = build_critic_wgan()

    # Define the optimizers for the generator and critic
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

    @tf.function
    def train_step(images):
        for i in range(critic_steps):
            # Generate random noise as input to the generator
            noise = tf.random.normal([batch_size, 75*75*2])
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
                total_loss = critic_loss_value + 10.0 * gp

            # Compute the gradients and update the critic's parameters
            critic_gradients = critic_tape.gradient(total_loss, critic.trainable_variables)
            critic_optimizer.apply_gradients(zip(critic_gradients, critic.trainable_variables))

        # Generate random noise as input to the generator
        noise = tf.random.normal([batch_size, 75*75*2])

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
    trained_generator, generator_loss = train_wgan(X_train_array, EPOCHS, BATCH_SIZE, STEPS)
    return trained_generator, generator_loss