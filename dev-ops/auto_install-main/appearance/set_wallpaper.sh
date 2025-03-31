#!/bin/bash

# Проверка наличия установленной утилиты gsettings
if ! command -v gsettings &>/dev/null; then
  echo "Утилита 'gsettings' не найдена. Убедитесь, что вы используете среду рабочего стола GNOME."
  exit 1
fi

# URL-адрес картинки для скачивания
image_url="https://r4.wallpaperflare.com/wallpaper/317/135/758/desaturated-desert-windows-10-windows-10x-windows-logo-hd-wallpaper-b37bc3ad8dd90b457fa692e778abb828.jpg"

# Путь для сохранения скачанной картинки
image_path="$HOME/Pictures/wallpaper.jpg"

# Скачивание картинки
wget -O "$image_path" "$image_url"
if [ $? -ne 0 ]; then
  echo "Не удалось скачать картинку."
  exit 1
fi

# Задание картинки в качестве обоев рабочего стола
gsettings set org.gnome.desktop.background picture-uri "file://$image_path"

echo "Картинка успешно установлена на рабочий стол."
