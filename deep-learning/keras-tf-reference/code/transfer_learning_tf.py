# MobileNetV2 / TF-Hub transfer learning walkthrough on cats_vs_dogs:
# (1) use a full ImageNet classifier head as-is, (2) swap in a feature
# extractor and train a new small head on top, (3) save the result.
#
# Merged from the archive's transferLearning.py (fuller walkthrough - includes
# the ImageNet-classifier demo section) and saveTransfLearning.py (a near-
# duplicate that explicitly freezes the feature extractor and saves the
# trained model at the end). Uses transferLearning.py as the base and folds
# in saveTransfLearning.py's two improvements:
#   - `feature_extractor.trainable = False` (the original transferLearning.py
#     never explicitly freezes the extractor before training the new head -
#     relying on hub.KerasLayer's default, which happens to already be
#     non-trainable, but leaving it implicit is a latent bug waiting to
#     happen if that default ever changes or the layer is swapped out)
#   - saving the trained model at the end

# %%
import time
import numpy as np
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras import layers

import PIL.Image as Image
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

IMAGE_RES = 224

# %% Part 1: use a full pretrained ImageNet classifier as-is (no training)
CLASSIFIER_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
classifier_model = tf.keras.Sequential([
    hub.KerasLayer(CLASSIFIER_URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))
])

grace_hopper = tf.keras.utils.get_file(
    'image.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize((IMAGE_RES, IMAGE_RES))
grace_hopper = np.array(grace_hopper) / 255.0

result = classifier_model.predict(grace_hopper[np.newaxis, ...])
predicted_class = np.argmax(result[0], axis=-1)

labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())
print("Prediction:", imagenet_labels[predicted_class])

# %% Load cats_vs_dogs for the transfer-learning part
(train_examples, validation_examples), info = tfds.load(
    'cats_vs_dogs', with_info=True, as_supervised=True,
    split=['train[:80%]', 'train[80%:]'],
)
num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes


def format_image(image, label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES)) / 255.0
    return image, label


BATCH_SIZE = 32
train_batches = train_examples.cache().shuffle(num_examples // 4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_examples.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)

# %% Part 2: feature extraction - freeze the pretrained backbone, train only a new head
FEATURE_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(FEATURE_URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))
feature_extractor.trainable = False  # explicit freeze - don't rely on the layer's default

model = tf.keras.Sequential([
    feature_extractor,
    layers.Dense(num_classes)
])
model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

EPOCHS = 6
history = model.fit(train_batches, epochs=EPOCHS, validation_data=validation_batches)

# %% Inspect predictions
class_names = np.array(info.features['label'].names)
image_batch, label_batch = next(iter(train_batches.take(1)))
image_batch = image_batch.numpy()
label_batch = label_batch.numpy()

predicted_batch = model.predict(image_batch)
predicted_ids = np.argmax(tf.squeeze(predicted_batch).numpy(), axis=-1)
predicted_class_names = class_names[predicted_ids]

plt.figure(figsize=(10, 9))
for n in range(min(30, len(image_batch))):
    plt.subplot(6, 5, n + 1)
    plt.subplots_adjust(hspace=0.3)
    plt.imshow(image_batch[n])
    color = "blue" if predicted_ids[n] == label_batch[n] else "red"
    plt.title(predicted_class_names[n].title(), color=color)
    plt.axis('off')
_ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")
plt.show()

# %% Save the trained model
t = time.time()
export_path_keras = "./{}.h5".format(int(t))
print(export_path_keras)
model.save(export_path_keras)
