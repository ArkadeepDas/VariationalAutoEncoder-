# During training process mean and standard deviation get updated and z is random sample
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load Data set
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train/255
x_test = x_test/255

image_width = 28
image_height = 28
num_channels = 1

x_train = x_train.reshape(x_train.shape[0], image_height, image_width, num_channels)
x_test = x_test.reshape(x_test.shape[0], image_height, image_width, num_channels)
input_shape = (image_height, image_width, num_channels)

batch_size = 32

latent_dim = 2 

encoder_input_layer = tf.keras.layers.Input(shape = input_shape, name = 'encoder_input')
x = tf.keras.layers.Conv2D(32, (3, 3), padding = 'same', activation = 'relu')(encoder_input_layer)
x = tf.keras.layers.Conv2D(64, (3, 3), padding = 'same', strides = (2, 2), activation = 'relu')(x)
x = tf.keras.layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu')(x)
x = tf.keras.layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu')(x)

# Now we have to define mean varience and z
# Lets save the convolution shape
conv_shape = tf.keras.backend.int_shape(x) #14x14x64
print(conv_shape)
# This shape of conv have to provide to decoder
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(32, activation = 'relu')(x)

# Now define mean and standard daviation
# These are the distributions
z_mu = tf.keras.layers.Dense(latent_dim, name = 'latent_mu')(x)
z_std = tf.keras.layers.Dense(latent_dim, name = 'latent_std')(x)

# Now comes to the reparameterization technique. How to create z
# Define a sampling function from the distribution
def sample_z(args):
    z_mu, z_std = args
    eps = tf.keras.backend.random_normal(shape = (tf.keras.backend.shape(z_mu)[0], tf.keras.backend.int_shape(z_mu)[1]))
    sample = z_mu + tf.keras.backend.exp(z_std / 2) * eps
    return sample
# eps = tf.keras.backend.random_normal(shape = (tf.keras.backend.shape(z_mu)[0], tf.keras.backend.int_shape(z_mu)[1]))
# print(eps)

# Now we have to sample from latent vector ort mean and standard daviation
# It is the last layer of the encoder
# Lambda layers are useful when you need to do some operations on the previous layer but don't want to add any trainable weight to it.
z = tf.keras.layers.Lambda(sample_z, output_shape = (latent_dim), name = 'z')([z_mu, z_std])

# Now we have to define and summarize encoder
encoder = tf.keras.models.Model(encoder_input_layer, [z_mu, z_std, z], name='encoder')
print(encoder.summary())

# Now the decoder part
# It takes latent vector(Random samples)
decoder_input_layer = tf.keras.layers.Input(shape = (latent_dim), name = 'decoder_input')
# print(tf.keras.backend.int_shape(decoder_input_layer))
# Need to start with a shape that can be remapped to original image shape.
# So we need to rebuild the image in decoder
x = tf.keras.layers.Dense(conv_shape[1]*conv_shape[2]*conv_shape[3], activation = 'relu')(decoder_input_layer)
x = tf.keras.layers.Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
x = tf.keras.layers.Conv2DTranspose(32, (3, 3), padding = 'same', activation = 'relu', strides = (2, 2))(x)
x = tf.keras.layers.Conv2D(32, (3, 3), padding = 'same', activation = 'relu')(x)
x = tf.keras.layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu')(x)
x = tf.keras.layers.Conv2D(num_channels, (3, 3), padding = 'same', activation = 'sigmoid', name = 'decoder_output')(x)

# Now we have to define and summarize decoder
decoder = tf.keras.models.Model(decoder_input_layer, x, name = 'decoder')
print(decoder.summary())

# Now we apply the decoder to the Z so that we can get output image
z_decoded = decoder(z)

# Now the main part loss function
# Loss function have two part 1) reconstruction loss 2) KL divergence
def loss(x, z_decode):
    mu , std, z = encoder(x)
    z_decode = decoder(z)
    x = tf.keras.backend.flatten(x)
    z_decode = tf.keras.backend.flatten(z_decode)
    re_con_loss = tf.keras.losses.binary_crossentropy(x, z_decode)
    kl_loss = -5e-4 * tf.keras.backend.mean(1 + std - tf.keras.backend.square(mu) - tf.keras.backend.exp(std), axis=-1)
    return tf.keras.backend.mean(re_con_loss + kl_loss)

vae = tf.keras.models.Model(encoder_input_layer, z_decoded)
vae.compile(optimizer = 'adam', loss = loss)
print(vae.summary())

vae.fit(x = x_train, y = x_train, epochs = 5, batch_size = batch_size, validation_split = 0.2)

# The output of the encoder is mean standard daviation and samples
mu, _, _ = encoder.predict(x_test)

plt.scatter(mu[:, 0], mu[:, 1], c = y_test, cmap = 'brg')
plt.xlabel('dim1')
plt.ylabel('dim2')
plt.colorbar()
plt.show()

# From the sample space we rebuild the image
sample_vector = np.array([[0, 4]])
decoder_example = decoder.predict(sample_vector)
decoder_example_output = decoder_example.reshape(image_height, image_width) * 255
cv2.imwrite("demo.jpg", decoder_example_output)