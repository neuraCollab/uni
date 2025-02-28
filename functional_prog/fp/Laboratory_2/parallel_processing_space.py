import cv2
import numpy as np
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import tkinter as tk
from tkinter import filedialog, messagebox

# Функции фильтрации
def apply_sharpen(image):
    try:
        # Фильтр увеличения резкости (с использованием свёртки)
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened, "sharpen"
    except Exception as e:
        print(f"Error in apply_sharpen: {e}")
        return None, "sharpen"

def apply_sepia(image):
    try:
        # Фильтр сепии (цветовая матрица)
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        sepia = cv2.transform(image, sepia_filter)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)
        return sepia, "sepia"
    except Exception as e:
        print(f"Error in apply_sepia: {e}")
        return None, "sepia"

def apply_resize(image, scale=0.5):
    try:
        # Уменьшение размера изображения (масштабирование)
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return resized, "resize"
    except Exception as e:
        print(f"Error in apply_resize: {e}")
        return None, "resize"

# Обработка одного изображения: параллельное применение фильтров
def process_image(image_path, output_folder):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Не удалось загрузить изображение {image_path}")
            return

        # Создаём отдельную папку для выходных файлов изображения или сохраняем с суффиксом
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image_output_dir = os.path.join(output_folder, base_name)
        os.makedirs(image_output_dir, exist_ok=True)

        results = {}
        # Используем потоки для параллельного применения фильтров
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(apply_sharpen, image): "sharpen",
                executor.submit(apply_sepia, image): "sepia",
                executor.submit(apply_resize, image): "resize"
            }
            for future in as_completed(futures):
                filter_name = futures[future]
                processed_img, suffix = future.result()
                if processed_img is not None:
                    results[suffix] = processed_img
                else:
                    print(f"Ошибка обработки фильтром: {filter_name}")

        # Сохранение результатов. Каждый фильтр сохраняется под своим именем
        for suffix, proc_img in results.items():
            output_path = os.path.join(image_output_dir, f"{base_name}_{suffix}.png")
            cv2.imwrite(output_path, proc_img)
            print(f"Сохранено: {output_path}")

    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {e}")

# Обработка всех изображений из выбранной папки с использованием multiprocessing
def process_all_images(input_directory, output_directory):
    if not os.path.isdir(input_directory):
        print("Некорректная входная директория")
        return

    os.makedirs(output_directory, exist_ok=True)
    image_files = [os.path.join(input_directory, f) for f in os.listdir(input_directory)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("Изображения не найдены в директории")
        return

    # Пул процессов для обработки изображений параллельно
    with multiprocessing.Pool(processes=min(len(image_files), multiprocessing.cpu_count())) as pool:
        tasks = []
        for image_path in image_files:
            tasks.append(pool.apply_async(process_image, args=(image_path, output_directory)))
        # Ожидаем завершения всех процессов
        for task in tasks:
            task.wait()

    print("Обработка изображений завершена.")

# Интерфейс с использованием tkinter
def choose_directory(title):
    directory = filedialog.askdirectory(title=title)
    return directory

def start_processing():
    input_dir = input_directory.get()
    output_dir = output_directory.get()
    if not input_dir or not output_dir:
        messagebox.showwarning("Внимание", "Необходимо выбрать входную и выходную директории")
        return

    try:
        process_all_images(input_dir, output_dir)
        messagebox.showinfo("Готово", f"Изображения обработаны и сохранены в {output_dir}")
    except Exception as e:
        messagebox.showerror("Ошибка", f"Произошла ошибка при обработке: {e}")

def select_input_directory():
    dir_path = choose_directory("Выберите папку с исходными изображениями")
    if dir_path:
        input_directory.set(dir_path)
        lbl_input.config(text=f"Вход: {dir_path}")

def select_output_directory():
    dir_path = choose_directory("Выберите папку для сохранения обработанных изображений")
    if dir_path:
        output_directory.set(dir_path)
        lbl_output.config(text=f"Выход: {dir_path}")

# Создание графического интерфейса
def create_interface():
    root = tk.Tk()
    root.title("Параллельная обработка изображений")

    global input_directory, output_directory, lbl_input, lbl_output
    input_directory = tk.StringVar()
    output_directory = tk.StringVar()

    frame = tk.Frame(root, padx=10, pady=10)
    frame.pack()

    btn_select_input = tk.Button(frame, text="Выбрать исходную папку", command=select_input_directory)
    btn_select_input.grid(row=0, column=0, padx=5, pady=5)

    lbl_input = tk.Label(frame, text="Вход: не выбран")
    lbl_input.grid(row=0, column=1, padx=5, pady=5)

    btn_select_output = tk.Button(frame, text="Выбрать папку для сохранения", command=select_output_directory)
    btn_select_output.grid(row=1, column=0, padx=5, pady=5)

    lbl_output = tk.Label(frame, text="Выход: не выбран")
    lbl_output.grid(row=1, column=1, padx=5, pady=5)

    btn_start = tk.Button(frame, text="Начать обработку", command=start_processing)
    btn_start.grid(row=2, column=0, columnspan=2, padx=5, pady=10)

    return root

if __name__ == "__main__":
    root = create_interface()
    root.mainloop()
