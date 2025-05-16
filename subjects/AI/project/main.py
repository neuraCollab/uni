# %%

import mne
from sklearn.svm import SVC
from pathlib import Path
from sklearn.model_selection  import train_test_split
import numpy as np
# Загружаем пример EEG датасета
eeg_data = Path(mne.datasets.sample.data_path())  # Путь как объект Path

# Преобразуем в строку или используем Path для объединения путей
raw = mne.io.read_raw_fif(eeg_data / 'MEG/sample/sample_audvis_raw.fif', preload=True)

# Предобработка данных
raw.filter(1, 40)  # Фильтрация сигнала в диапазоне от 1 до 40 Гц

# %%
raw.plot(duration=20, n_channels=30, scalings='auto', title="Raw EEG data")

# Сегментация на эпизоды по событиям
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, event_id=None, tmin=-0.2, tmax=0.5, baseline=(None, 0), detrend=1, preload=True)

# Визуализация эпизодов
epochs.plot()

X = epochs.get_data()  # Признаки

# Преобразование в 2D для классификации
X = X.reshape(X.shape[0], -1)

# Моделирование с использованием SVM
y = epochs.events[:, -1]  # Метки классов
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# %%

# Оценка модели
accuracy = clf.score(X_test, y_test)
print(f'Accuracy: {accuracy}')

# %%
