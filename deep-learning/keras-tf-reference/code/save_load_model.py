# Keras model persistence patterns: checkpoint callback during training,
# manual save_weights/load_weights, SavedModel format, and HDF5 format.
# Ported from the archive's saveAndLoadModel.py.
#
# BUG FIX: the original had `os.system('!mkdir - p saved_model')` before
# `model.save('saved_model/my_model')`. That's broken shell syntax - a
# leftover Jupyter `!mkdir -p` cell magic that got wrapped in os.system()
# incorrectly (the `!` and the space in `- p` are not valid for a real shell
# command, and this would either error or no-op depending on the OS/shell).
# It's also unnecessary: model.save() creates any missing directories itself.
# The line is simply removed below.

# %%
import os
import tensorflow as tf
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.metrics.SparseCategoricalAccuracy()])
    return model


# %% 1. ModelCheckpoint callback - saves weights during training
model = create_model()
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                   save_weights_only=True,
                                                   verbose=1)
model.fit(train_images, train_labels, epochs=10,
          validation_data=(test_images, test_labels), callbacks=[cp_callback])

# reload into a fresh, untrained model instance
model = create_model()
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# %% 2. Periodic checkpoints (every N steps), keeping the epoch in the filename
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
batch_size = 32

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, verbose=1, save_weights_only=True,
    save_freq=5 * batch_size)

model = create_model()
model.save_weights(checkpoint_path.format(epoch=0))
model.fit(train_images, train_labels, epochs=50, batch_size=batch_size,
          callbacks=[cp_callback], validation_data=(test_images, test_labels), verbose=0)

latest = tf.train.latest_checkpoint(checkpoint_dir)
model = create_model()
model.load_weights(latest)

# %% 3. Manual save_weights/load_weights (no callback)
model.save_weights('./checkpoints/my_checkpoint')
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')

# %% 4. SavedModel format (TF's native serialization - includes architecture,
# weights, and optimizer state)
model = create_model()
model.fit(train_images, train_labels, epochs=5)
model.save('saved_model/my_model')  # creates the directory itself - no manual mkdir needed
new_model = tf.keras.models.load_model('saved_model/my_model')
new_model.summary()

# %% 5. HDF5 format (single-file .h5, framework-portable)
model = create_model()
model.fit(train_images, train_labels, epochs=5)
model.save('my_model.h5')
new_model = tf.keras.models.load_model('my_model.h5')
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
