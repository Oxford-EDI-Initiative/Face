import tensorflow as tf
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow.keras import layers #type: ignore
from tensorflow.keras import Sequential # type: ignore


#defining a method to get the cross-entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def make_generator():
    #these stride values for upsampling only work when images are 180x180 (defined in img_height, img_width below)
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(5000,)))

    #initial FC layer to upsample the initial random vector
    model.add(layers.Dense(9*9*256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    #reshaping into a 9x9 image with 256 channels
    model.add(layers.Reshape((9, 9, 256)))

    #upsampling and reducing the number of channels
    model.add(layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    #upsampling and reducing the number of channels
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(5, 5), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    #upsampling and reducing the number of channels
    model.add(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    #finally upsampling to produce an image of the correct size (180x180x3 for RGB)
    model.add(layers.Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    model.add(layers.Rescaling(scale=127.5, offset=127.5))

    return model

def make_discriminator():
    classifier = Sequential([
        layers.Input(shape=(180, 180, 3)),
        layers.Rescaling(scale=1/127.5, offset=-1.0),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)])
    return classifier

def generator_loss(fake_output):
    #comparing the classification made by the discriminator on the artificial images (fake_output) to all 1's
    #if the generator is good, you would expect the discriminator to predict all 1's (real)
    loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    return loss

def discriminator_loss(fake_output, real_output):
    #comparing the classification of the real/fake images to all 1's/0's as the discriminator should predict all 1's/0's
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

@tf.function #apparently optimises the function as a graph
def train_step(real_images, discriminator, generator, noise_dim, batch_size, gen_optimizer, disc_optimizer):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(real_images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, num_epochs, discriminator, generator, checkpoint, checkpoint_prefix, noise_dim, batch_size, gen_optimizer, disc_optimizer):
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch}')
        i = 0
        for image_batch, _ in dataset:
            print(i)
            i += 1
            train_step(image_batch, discriminator, generator, noise_dim, batch_size, gen_optimizer, disc_optimizer)
        #showing the images currently being generated
        test_input = tf.random.normal([1, noise_dim])
        predictions = generator(test_input, training=False)[0]._numpy().astype(np.int32)
        print(predictions)
        plt.imshow(predictions)
        plt.show()
        #saving the state of training and models learned every 10 epochs
        if epoch % 10 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

def get_data(train_path, val_path, batch_size, num_batches, img_height, img_width):
    #getting the training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
    #taking only a few batches to speed up training (just for testing)
    train_ds = train_ds.take(num_batches)

    #getting the validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
    val_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width), 
    batch_size=batch_size)
    #again, taking only a few batches to speed up training
    val_ds = val_ds.take(num_batches)

    return train_ds, val_ds


#creating datasets and splitting into training and test data
#file paths to the training and validation data 
train_path = Path('Datasets/split_class_images/train') 
val_path = Path('Datasets/split_class_images/val')

#setting parameters for the images
batch_size = 128
num_batches = 25
img_height = 180
img_width = 180
noise_dim = 5000

"""
generator = make_generator()
test_input = tf.random.normal([1, noise_dim])
predictions = generator(test_input, training=False)
print(predictions)
print(generator.get_weights())
plt.imshow(predictions[0], vmin=0, vmax=255)
plt.show()
"""


#getting the data
train_ds, val_ds = get_data(train_path, val_path, batch_size, num_batches, img_height, img_width)

#setting the optimizers
disc_optimizer = tf.keras.optimizers.Adam(1e-3)
gen_optimizer = tf.keras.optimizers.Adam(1e-3)

#making the models
discriminator = make_discriminator()
generator = make_generator()

#setting up the checkpoints to save the model in case training is interrupted
checkpoint_dir = './GAN Training Checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
                                 discriminator_optimizer=disc_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

#training the model
train(train_ds, 10, discriminator, generator, checkpoint, checkpoint_prefix, noise_dim, batch_size, gen_optimizer, disc_optimizer)
