import pyshark

# Укажите путь к файлу pcapng, который нужно прочитать
file_path = "./ethernet_logs.pcapng"

# Открываем файл pcapng для чтения
capture = pyshark.FileCapture(file_path)

# Проходимся по каждому пакету в файле
for packet in capture:
    # Выводим информацию о пакете
    print("Packet:")
    print(f"  Time: {packet.sniff_timestamp}")
    print(f"  Source IP: {packet.ip.src}")
    print(f"  Destination IP: {packet.ip.dst}")
    # и другие поля, которые вам интересны

# Закрываем файл pcapng
capture.close()
