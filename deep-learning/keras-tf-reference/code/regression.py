# Keras regression pipeline on the UCI Auto MPG dataset: normalization,
# a single-feature linear model, a multi-feature linear model, and a small
# DNN regressor. Trimmed from the archive's regression.py (originally
# adapted from https://www.tensorflow.org/tutorials/keras/regression) down
# to the essentials - dropped most of the plotting/EDA cells, kept the
# modeling pattern (Normalization layer -> Sequential -> compile -> fit -> evaluate).

# %%
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import pandas as pd

# %% Load + clean data
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names, na_values='?',
                           comment='\t', sep=' ', skipinitialspace=True)

dataset = raw_dataset.dropna().copy()
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()
train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

# %% Normalization layer learns feature-wise mean/variance from training data
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

# %% Single-feature linear regression (horsepower -> MPG)
horsepower = np.array(train_features['Horsepower'])
horsepower_normalizer = layers.Normalization(input_shape=[1, ], axis=None)
horsepower_normalizer.adapt(horsepower)

horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])
horsepower_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1),
                          loss='mean_absolute_error')
horsepower_model.fit(train_features['Horsepower'], train_labels,
                      epochs=100, verbose=0, validation_split=0.2)

# %% Multi-feature linear regression
linear_model = tf.keras.Sequential([normalizer, layers.Dense(units=1)])
linear_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1),
                      loss='mean_absolute_error')
linear_model.fit(train_features, train_labels, epochs=100, verbose=0,
                  validation_split=0.2)

# %% Small DNN regressor
def build_and_compile_model(norm):
    model = tf.keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
    return model

dnn_model = build_and_compile_model(normalizer)
dnn_model.fit(train_features, train_labels, validation_split=0.2, verbose=0, epochs=100)

test_results = {
    'horsepower_model': horsepower_model.evaluate(test_features['Horsepower'], test_labels, verbose=0),
    'linear_model': linear_model.evaluate(test_features, test_labels, verbose=0),
    'dnn_model': dnn_model.evaluate(test_features, test_labels, verbose=0),
}
print(pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T)

# %% Save / reload
dnn_model.save('dnn_model')
reloaded = tf.keras.models.load_model('dnn_model')
print(reloaded.evaluate(test_features, test_labels, verbose=0))
