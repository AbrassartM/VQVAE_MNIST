import tensorflow as tf 
import numpy as np

def dataset(params):

    train_size = params['train_size']
    batch_size = params['batch_size']
    test_size = params['test_size']
    data_type = params['data_type']
    vae_type = params['vae_type']
    
    def preprocess_images_CVAE(images):

        images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
        return np.where(images > .5, 1.0, 0.0).astype('float32')
    
    def preprocess_images_VQVAE(train_images, test_images):
        x_train = np.expand_dims(train_images, -1)
        x_test = np.expand_dims(test_images, -1)
        x_train_scaled = (x_train / 255.0) - 0.5
        x_test_scaled = (x_test / 255.0) - 0.5

        data_variance = np.var(x_train / 255.0)

        return x_train_scaled, x_test_scaled, data_variance
        
    
    def prepare_dataset_CVAE(train_images, test_images):

        train_images = preprocess_images_CVAE(train_images)
        test_images = preprocess_images_CVAE(test_images)

        train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                        .shuffle(train_size).batch(batch_size))
        test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                        .shuffle(test_size).batch(batch_size))
        
        return train_dataset, test_dataset
    
    if data_type == 'MNIST':
        train_images, test_images = extract_MNIST()
    elif data_type == 'FASHION_MNIST':
        train_images, test_images = extract_fashionMNIST()
    else:
        raise NotImplementedError('dataset %s not implemented' %data_type)
    
    if vae_type == "CVAE":
        train_dataset, test_dataset = prepare_dataset_CVAE(train_images, test_images)
        data_variance = None
        test_images = None

        return train_dataset, test_dataset, data_variance, test_images
        
    elif vae_type == "VQVAE":
        train_dataset, test_dataset, data_variance = preprocess_images_VQVAE(train_images, test_images)
        return train_dataset, test_dataset, data_variance, test_images

def extract_MNIST():
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    return train_images, test_images

def extract_fashionMNIST():
    (train_images, _), (test_images, _) = tf.keras.datasets.fashion_mnist.load_data()
    return train_images, test_images

