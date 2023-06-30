from IPython import display

from VAE_MNIST import CVAE
from VQVAE_trainer import VQVAETrainer
from dataset_MNIST import dataset
from utils import compute_loss, generate_and_save_images, show_subplot
import tensorflow as tf
import numpy as np
import time, json
from argparse import ArgumentParser

parser = ArgumentParser()

# parser.add_argument('-save_path', default='saved_images', type=str,
#                     help='path to save images')
parser.add_argument('-params', default='parameters.json', type=str, dest='parameter_path',
                    help='path to parameters file')

args = parser.parse_args()


with open(args.parameter_path) as file:
    parameters = json.load(file)

test_dataset, train_dataset, data_variance, test_images = dataset(parameters)

optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)             
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

random_vector_for_generation = tf.random.normal(
    shape=[parameters['num_examples_to_generate'], parameters['latent_dim']]
)

model = CVAE(parameters['latent_dim'])

assert parameters['batch_size'] >= parameters['num_examples_to_generate']

if parameters['vae_type'] == "CVAE":
    for test_batch in test_dataset.take(1):
        test_sample = test_batch[0:parameters['num_examples_to_generate'], :, :, :]
    for epoch in range(1, parameters['epochs'] + 1):
        start_time = time.time()
        for train_x in train_dataset:
            train_step(model, train_x, optimizer)
        end_time = time.time()

        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            loss(compute_loss(model, test_x))
        elbo = -loss.result()
        display.clear_output(wait=False)
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
            .format(epoch, elbo, end_time - start_time))
        
    generate_and_save_images(model, epoch, test_sample, parameters['data_type'])

        
elif parameters['vae_type'] == "VQVAE":
    vqvae_trainer = VQVAETrainer(data_variance, latent_dim=16, num_embeddings=128)
    vqvae_trainer.compile(optimizer)
    vqvae_trainer.fit(train_dataset, epochs=30, batch_size=128)      

    test_images_scaled = (np.expand_dims(test_images, -1) / 255.0) - 0.5
    trained_vqvae_model = vqvae_trainer.vqvae
    idx = np.random.choice(len(test_images_scaled), 10)
    test_images = test_images_scaled[idx]
    reconstructions_test = trained_vqvae_model.predict(test_images)

    for test_image, reconstructed_image in zip(test_images, reconstructions_test):
        show_subplot(test_image, reconstructed_image)