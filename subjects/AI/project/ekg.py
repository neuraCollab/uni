import os
import zipfile
import wfdb

# Распаковка архива
zip_path = 'ptb-diagnostic-ecg-database-1.0.0.zip'
extract_dir = 'ptb_ecg'
if not os.path.exists(extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

# Поиск всех записей
records = []
for root, _, files in os.walk(extract_dir):
    for file in files:
        if file.endswith('.hea'):
            rec_path = os.path.splitext(os.path.join(root, file))[0]
            records.append(rec_path)

# Пример анализа одной записи
record = wfdb.rdrecord(records[0])
print(f"Запись: {records[0]}")
print(f"Сигналы: {record.sig_name}")
print(f"Частота дискретизации: {record.fs}")
print(f"Данные сигнала (shape): {record.p_signal.shape}")
print(f"Комментарий: {record.comments}")

# Для анализа всех записей — пройдитесь по списку records
