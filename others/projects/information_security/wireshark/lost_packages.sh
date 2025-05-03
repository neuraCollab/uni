#!/bin/bash

input_file="input.pcapng"  # Имя входного файла pcapng

# Используем tshark для анализа потерь пакетов
tshark -r "$input_file" -qz io,stat,1,"COUNT(frame) frame.len < 64"
