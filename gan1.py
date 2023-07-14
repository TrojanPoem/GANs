"""
Important notes:

a custom loss function in keras is defined with

def loss(y_pred, y_true)

y_true: feed by us into model.fit(x_train, y_train) => y_train = y_true
y_pred: calculated by keras by feeding x_train into the model and finding the output


The discriminator loss function:


D_real_loss = -log(D(real_img)) => when real images

D_fake_loss = - log(1- D(fake_image)) => when fake images

this is the normal binary cross entropy loss:

loss = -y_true log (y_pred) - (1-y_true) * log(1- y_pred)

For real labels (value = 1):
-----------------------------------
loss = -log(y_pred) - (1-1) * log(1-y_pred) = -log(y_pred) => same as D_real_loss

For fake labels(value = 0):
--------------------------
loss = -0*log(y_pred) - (1-0) * log(1-y_pred) = -log(1-y_pred) => same as D_fake_loss


=>>> No need to define a custom loss function.


For the generator:

Do not minimize log(1-D(G(z)), it has low gradient => slow training.

Use min -log(D(G(z))) instead.

This is the same as binary cross entropy so no need for a custom loss function.

For the generator images , we want the discriminator to produce 1, so the output is real = 1

loss = -y_true * log(y_pred) - (1-y_true) * log(1-y_pred)

loss = -log(y_pred) - (1-1) * log(1-y_pred)

loss = -log(y_pred) => the required loss function.


Notes:
------
#To train the discriminator => The discriminator model is used.

#To train the generator => use a combinaed model to get D(G(z)) => when evaluated by keras ,  z : noise (latent space input) , fed into generator G(z) , then fed into discriminator D(G(z)) to get 1 , which means we tricked the disciminator.

#This generator does not have real images in its loss, so it just learns to trick the discriminator and can generate meaningless images.

#Draw the latent space to get an idea of regions of meaningful images.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

#Constants
batch_size = 50
latent_space_size = 100
image_size = 784          # 28 x 28
max_epochs = 400

#Build the generator model
#The input is the latent space vector (1x100) and the output is the generated image (784)
generator = tf.keras.Sequential([

    tf.keras.layers.Dense(units = 256, input_dim = latent_space_size, activation = 'relu'),
    tf.keras.layers.Dense(units = image_size, activation = 'sigmoid') #output an image of image_size

])

#Build the discriminator model
discriminator = tf.keras.Sequential([

    tf.keras.layers.Dense(units = 256, input_dim = image_size, activation = 'relu'), #input is the image of image_size
    tf.keras.layers.Dense(units = 1, activation = 'sigmoid') #Output the probability of being real: => 1:real , 0: fake

])


#The groundtruth is the discriminator believing that the generated images are real hence: x_train = noise,  y_train = real_labels
#The combined model will be used to train the generator.
#The input to that model is the latent sapace vector, while the output of it is the label real: 1 , fake: 0


#The combined model input
DG_in = tf.keras.layers.Input(shape = (latent_space_size,)) #The input to the generator is a vector f 100D (latent space size)

#Feed the input to the generator to get the output image:

g_out = generator(DG_in)

#Prevent the discriminaotr from training
discriminator.trainable = False

#Feeding the discriminaor with generator output to get the classification: fake/real (in this case fake)
d_fake = discriminator(g_out)

#Build the combined model of DG for training of fake.
DG_model = tf.keras.Model(inputs = DG_in, outputs = d_fake)


#Compile the models

discriminator.compile(optimizer = 'adam', loss = 'binary_crossentropy')
DG_model.compile(optimizer = 'adam', loss = 'binary_crossentropy')


# Load the MNIST dataset
(x_mnist, y_mnist), (_, _) = mnist.load_data()

#Show a sample image
plt.figure()
plt.imshow(x_mnist[50000,:,:], 'gray') #3
plt.show()

#Create the labels for real and fake based on the batch size (50)
fake_label = np.zeros(batch_size)
real_label = np.ones(batch_size)

#Train the models
for epoch in range(max_epochs):

    #----------------------------Training of discriminator------------------------------------------------------------#

    #Generate noise as the latent space input
    noise = np.random.normal(size = (batch_size, latent_space_size)) #Each row represent an input of the latent space.

    #Get fake training batch
    fake_imgs = generator.predict(noise)


    #Get real training batch
    idx = np.random.randint(low = 0, high = 60000, size = batch_size)

    real_imgs =  x_mnist[idx,:,:].reshape(-1, 784)

    #Train on real images
    d_loss_real = discriminator.train_on_batch(real_imgs, real_label)

    #Train on fake images
    d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_label)

    #Total loss for printing
    d_loss = d_loss_real + d_loss_fake

     #----------------------------Training of Generator------------------------------------------------------------#

    #The groundtruth is the discriminator believing that this images are real , x_train = noise,  y_train = real_labels

    #Generate noise as the latent space input
    noise = np.random.normal(size = (batch_size, latent_space_size)) #Each row represent an input of the latent space.

    #Train the combind generator-discriminator model
    g_loss = DG_model.train_on_batch(noise, real_label) #When the discriminaotr thinks that this generated images are real, we are done training.


    if epoch % 1000 == 0:
        print('[+] Epoch:{}\n\tDiscriminator Loss:{}\n\tGenerator Loss:{}\n\t'.format(epoch, d_loss, g_loss))
