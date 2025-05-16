# %%
import tarfile
import nibabel as nib  # Исправлено: nlb -> nib
import matplotlib.pyplot as plt
from pathlib import Path

# %%
folder_path = Path('./IXI-T1.tar')
data_folder = Path('./data')

# Проверка, существует ли папка 'data', если нет — создание
if not data_folder.exists():
    data_folder.mkdir()

# Проверка, что папка 'data' пуста
if not any(data_folder.iterdir()):
    with tarfile.open(folder_path, 'r') as tar:
        tar.extractall(data_folder)

# %%
img = nib.load(data_folder / "IXI002-Guys-0828-T1.nii.gz")  # Исправлено: nlb -> nib
data = img.get_fdata()
print(data)

# %%
import matplotlib.pyplot as plt

# Индексы центральных срезов по осям
z_index = data.shape[2] // 2
y_index = data.shape[1] // 2
x_index = data.shape[0] // 2

# Создание подграфиков
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Визуализация центрального среза по оси Z
axs[0].imshow(data[:, :, z_index], cmap='gray')
axs[0].set_title(f'Slice along Z-axis (Index: {z_index})')
axs[0].axis('off')  # Отключение осей для лучшего отображения

# Визуализация среза по оси Y
axs[1].imshow(data[:, y_index, :], cmap='gray')
axs[1].set_title(f'Slice along Y-axis (Index: {y_index})')
axs[1].axis('off')

# Визуализация среза по оси X
axs[2].imshow(data[x_index, :, :], cmap='gray')
axs[2].set_title(f'Slice along X-axis (Index: {x_index})')
axs[2].axis('off')

# Показать график
plt.tight_layout()
plt.show()
 