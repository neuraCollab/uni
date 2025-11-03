import os
import cv2
import numpy as np
import multiprocessing as mp
from openpyxl import Workbook
import tkinter as tk
from tkinter import filedialog, messagebox

# Функция обработки отдельного фрагмента изображения
def process_image_chunk(params):
    chunk, idx, img_filename, dx, dy = params
    # Преобразование в оттенки серого и размытие для последующего порогового преобразования
    gray_img = cv2.cvtColor(chunk, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, bin_img = cv2.threshold(blurred_img, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_regions = []
    centers_list = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2 + dx
        cy = y + h // 2 + dy
        brightness = np.sum(gray_img[y:y+h, x:x+w])
        region_type = classify_region(area, brightness)
        
        if region_type != "-":
            detected_regions.append({
                "img": img_filename,
                "segment": idx + 1,
                "coords": (cx, cy),
                "brightness": brightness,
                "area": int(area),
                "type": region_type
            })
            centers_list.append((cx - dx, cy - dy, max(w, h) // 2, region_type))
    
    return detected_regions, centers_list

# Функция классификации по площади и яркости
def classify_region(area, brightness):
    if area < 300 and brightness > 200:
        return "Star"
    elif area > 300 and brightness > 1000:
        return "A bright star"
    elif area > 300 and brightness < 200:
        return "Planet"
    else:
        return "Star"

# Обработка всех изображений из заданной папки
def process_directory(src_folder, excel_filepath, out_folder):
    combined_results = []
    
    for fname in os.listdir(src_folder):
        full_path = os.path.join(src_folder, fname)
        if full_path.lower().endswith(('.png', '.jpg')):
            chunk_results = process_single_image(full_path, out_folder)
            combined_results.extend(chunk_results)
    
    export_to_excel(combined_results, excel_filepath)
    print(f"Processing finished. Results saved to {excel_filepath}")

# Экспорт данных в Excel
def export_to_excel(data, excel_filepath):
    wb = Workbook()
    ws = wb.active
    ws.title = "Results"
    
    header = ['Image', 'Segment', 'Coordinates', 'Brightness', 'Area', 'Type']
    ws.append(header)
    
    for record in data:
        ws.append([
            record['img'],
            record['segment'],
            f"{record['coords'][0]}, {record['coords'][1]}",
            record['brightness'],
            record['area'],
            record['type']
        ])
    
    # Настройка ширины столбцов
    for col in ws.columns:
        col_width = max(len(str(cell.value)) for cell in col) + 2
        ws.column_dimensions[col[0].column_letter].width = col_width
    
    wb.save(excel_filepath)

# Обработка одного изображения: разбиение, параллельная обработка и сохранение результатов
def process_single_image(img_path, out_folder):
    image = cv2.imread(img_path)
    base_name = os.path.basename(img_path)
    
    # Создаем папку для сохранения обработанных фрагментов
    output_dir = os.path.join(out_folder, os.path.splitext(base_name)[0])
    os.makedirs(output_dir, exist_ok=True)
    
    segments = split_image(image, 1000)
    tasks = [(seg, seg_idx, base_name, offset_x, offset_y) for seg, offset_x, offset_y, seg_idx in segments]
    
    with mp.Pool(processes=16) as pool:
        results = pool.map(process_image_chunk, tasks)
    
    all_data = []
    for (data_chunk, centers), (seg, ox, oy, seg_idx) in zip(results, segments):
        all_data.extend(data_chunk)
        annotate_and_save(seg, seg_idx, base_name, output_dir, centers)
    return all_data

# Функция разбиения изображения на равные блоки
def split_image(image, block_size):
    height, width, _ = image.shape
    blocks = []
    idx = 0
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = image[y:y+block_size, x:x+block_size]
            if block.size > 0:
                blocks.append((block, x, y, idx))
            idx += 1
    return blocks

# Сохранение обработанного блока с нанесенными аннотациями
def annotate_and_save(block, block_idx, img_name, out_dir, centers):
    for (cx, cy, rad, rtype) in centers:
        extended_rad = int(rad * 1.5)
        if rtype == "Star":
            col = (255, 0, 0)
        elif rtype == "Planet":
            col = (0, 0, 255)
        elif rtype == "A bright star":
            col = (0, 255, 0)
        else:
            col = (0, 255, 255)
        cv2.circle(block, (cx, cy), extended_rad, col, 4)
    
    block_filename = f"{block_idx+1}.png"
    cv2.imwrite(os.path.join(out_dir, block_filename), block)

# Графический интерфейс (новый стиль)
def build_ui():
    global btn_run_analysis, lbl_source, lbl_destination
    window = tk.Tk()
    window.title("Data Analysis for Cosmic Images")
    window.configure(bg="#f0f8ff")
    window.geometry("500x200")
    
    # Верхняя рамка для выбора папки с изображениями
    top_frame = tk.Frame(window, bg="#f0f8ff")
    top_frame.pack(pady=10)
    lbl_source = tk.Label(top_frame, text="Source Folder: Not selected", bg="#f0f8ff", font=("Arial", 10))
    lbl_source.pack(side=tk.LEFT, padx=5)
    btn_select_src = tk.Button(top_frame, text="Select Source", command=select_source_folder, bg="#add8e6", font=("Arial", 10))
    btn_select_src.pack(side=tk.LEFT, padx=5)
    
    # Средняя рамка для выбора места сохранения Excel-файла
    mid_frame = tk.Frame(window, bg="#f0f8ff")
    mid_frame.pack(pady=10)
    lbl_destination = tk.Label(mid_frame, text="Destination: Not selected", bg="#f0f8ff", font=("Arial", 10))
    lbl_destination.pack(side=tk.LEFT, padx=5)
    btn_select_dest = tk.Button(mid_frame, text="Select Destination", command=select_destination_folder, bg="#add8e6", font=("Arial", 10))
    btn_select_dest.pack(side=tk.LEFT, padx=5)
    
    # Нижняя рамка для кнопки запуска анализа
    bot_frame = tk.Frame(window, bg="#f0f8ff")
    bot_frame.pack(pady=20)
    btn_run_analysis = tk.Button(bot_frame, text="Start Analysis", command=run_analysis, bg="#90ee90", font=("Arial", 12, "bold"))
    btn_run_analysis.pack()
    
    return window

def select_source_folder():
    global source_path
    path = filedialog.askdirectory()
    if path:
        source_path = path
        lbl_source.config(text=f"Source Folder: {source_path}")

def select_destination_folder():
    global excel_file_path
    dest = filedialog.askdirectory()
    if dest:
        excel_file_path = os.path.join(dest, "results.xlsx")
        lbl_destination.config(text=f"Destination: {excel_file_path}")

def run_analysis():
    global source_path, excel_file_path, output_images_dir
    if not source_path:
        source_path = "photo"
    if not excel_file_path:
        excel_file_path = os.path.join(os.getcwd(), "results.xlsx")
    output_images_dir = "processed_parts"
    os.makedirs(output_images_dir, exist_ok=True)
    
    process_directory(source_path, excel_file_path, output_images_dir)
    messagebox.showinfo("Analysis Complete", f"Results saved to {excel_file_path}")

if __name__ == "__main__":
    # Глобальные переменные для хранения путей
    source_path = ""
    excel_file_path = ""
    output_images_dir = ""
    
    ui = build_ui()
    ui.mainloop()
