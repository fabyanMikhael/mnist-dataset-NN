import tensorflow as tf
import numpy as np
from tensorflow import keras

# Load the dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
# Normalize the data
train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.models.load_model('model.tf')


from PIL import Image
test = np.asarray(Image.open('test.png').convert('L')) / 255.0
test = test.reshape(-1,28,28,1)
print(np.argmax(model.predict(test)[0]))