#!/bin/bash

input_file="input.pcapng"  # Имя входного файла pcapng
output_file="http_requests_info.txt"  # Имя файла для сохранения информации о HTTP запросах

# Используем tshark для фильтрации HTTP запросов и сохранения всей информации в отдельный файл
tshark -r "$input_file" -Y "http.request" -T fields -e frame.number -e ip.src -e ip.dst -e http.request.method -e http.request.uri -e http.host -e http.user_agent -e http.referer -e http.request.full_uri > "$output_file"

echo "Информация о HTTP запросах была извлечена и сохранена в файле: $output_file"