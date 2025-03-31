#!/bin/bash

input_file="input.pcapng"  # Имя входного файла pcapng
output_file="http_requests.txt"  # Имя файла для сохранения HTTP запросов

# Используем tshark для извлечения HTTP запросов и сохранения их в отдельный файл
tshark -r "$input_file" -Y "http.request" -T fields -e http.request.line > "$output_file"

echo "HTTP запросы были извлечены и сохранены в файле: $output_file"
