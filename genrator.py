import tensorflow as tf
from tensorflow.keras import layers

# Define the Generator model architecture
def build_generator():
    model = tf.keras.Sequential()
    
    # Add a Dense layer with 7*7*256 units
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Reshape the output into a 7x7x256 tensor
    model.add(layers.Reshape((7, 7, 256)))

    # Upsample to 14x14x128
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Upsample to 28x28x64
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Output layer, generate 28x28x1 image
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model

# Define the Discriminator model architecture
def build_discriminator():
    model = tf.keras.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # Flatten the output
    model.add(layers.Flatten())

    # Output layer, single unit for binary classification
    model.add(layers.Dense(1))

    return model


# Define the loss function for the GAN
def gan_loss(fake_output, real_output):
    # Binary cross-entropy loss for GAN
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Generator loss
    gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    # Discriminator loss
    disc_loss_real = cross_entropy(tf.ones_like(real_output), real_output)
    disc_loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
    disc_loss = disc_loss_real + disc_loss_fake

    return gen_loss, disc_loss

# Instantiate the generator and discriminator models
generator = build_generator()
discriminator = build_discriminator()

# Define the optimizer for the models
generator_optimizer = tf.keras.optimizers.Adam()
discriminator_optimizer = tf.keras.optimizers.Adam()

# Define the number of epochs and batch size
num_epochs = 10
batch_size = 32

# Training loop
for epoch in range(num_epochs):
    for batch in dataset:
        # Train the discriminator on real images
        real_images = get_real_images(batch)
        with tf.GradientTape() as disc_tape:
            real_output = discriminator(real_images, training=True)
            fake_images = generator(noise, training=True)
            fake_output = discriminator(fake_images, training=True)
            gen_loss, disc_loss = gan_loss(fake_output, real_output)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        # Train the generator to fool the discriminator
        with tf.GradientTape() as gen_tape:
            fake_images = generator(noise, training=True)
            fake_output = discriminator(fake_images, training=True)
            gen_loss, disc_loss = gan_loss(fake_output, real_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    # Print the progress and loss for each epoch
    print(f"Epoch {epoch+1}/{num_epochs} - Generator Loss: {gen_loss}, Discriminator Loss: {disc_loss}")

# Generate new images using the trained generator
new_images = generator.predict(noise)

# Save the generated images
for i, image in enumerate(new_images):
    image_path = f"generated_image_{i}.png"  # Provide a unique file name or path for each image
    tf.keras.preprocessing.image.save_img(image_path, image)

    # Alternatively, you can save the image using PIL library
    # image = Image.fromarray((255 * image).astype(np.uint8))
    # image.save(image_path)

    print(f"Saved generated image at: {image_path}")
