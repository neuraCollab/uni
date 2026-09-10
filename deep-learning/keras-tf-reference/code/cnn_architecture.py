# A basic CNN architecture for Fashion-MNIST: Conv2D/MaxPooling stack ->
# Flatten -> Dense classifier. Ported from the archive's imgConvolution.py.
#
# NOTE: this is ARCHITECTURE-ONLY, matching the source file - the model is
# defined but never compiled or trained (no model.compile/model.fit calls in
# the original). Kept as-is since the point of this reference file is the
# layer pattern (conv -> pool -> conv -> pool -> flatten -> dense), not a
# runnable training script. To actually train it, add e.g.:
#   model.compile(optimizer='adam',
#                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#                 metrics=['accuracy'])
#   model.fit(train_dataset.batch(32), epochs=5, validation_data=test_dataset.batch(32))

import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu,
                            input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.summary()
