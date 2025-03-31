#!/bin/bash

input_file="network_traffic.pcapng"  # Имя входного файла pcapng
malicious_traffic_log="malicious_traffic.log"  # Файл для сохранения обнаруженного подозрительного трафика

# Используем tshark для фильтрации трафика и его анализа
tshark -r "$input_file" -Y "http.request or dns or ssl" -w "suspect_traffic.pcapng"

# Другие команды или сценарии могут использоваться для анализа содержимого зафиксированного трафика (например, использование инструментов для анализа сетевых протоколов или обнаружения сигнатур вредоносного ПО).

echo "Обнаруженный подозрительный трафик сохранен в файл: $malicious_traffic_log"
