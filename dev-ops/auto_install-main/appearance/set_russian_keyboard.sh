#!/bin/bash

# Проверка наличия установленной утилиты gsettings
if ! command -v gsettings &>/dev/null; then
  echo "Утилита 'gsettings' не найдена. Убедитесь, что вы используете среду рабочего стола GNOME."
  exit 1
fi

# Добавить русскую раскладку
gsettings set org.gnome.desktop.input-sources sources "[('xkb', 'us'), ('xkb', 'ru')]"

echo "Русская раскладка клавиатуры успешно добавлена."