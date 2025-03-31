#!/bin/bash

# Проверка наличия Gnome Shell Extensions в системе
if ! gnome-extensions --version &>/dev/null; then
  echo "Утилита gnome-extensions не найдена. Убедитесь, что Gnome Shell Extensions установлены."
  exit 1
fi

# Установка утилиты chrome-gnome-shell (если не установлена)
if ! dpkg -s chrome-gnome-shell &>/dev/null; then
  sudo apt-get update
  sudo apt-get install chrome-gnome-shell
fi

# Проверка наличия настроек расширений Gnome Shell
if ! gnome-shell-extension-prefs &>/dev/null; then
  echo "Настройки расширений Gnome Shell не доступны."
  exit 1
fi

# Массив URL расширений
extensions=(
"https://extensions.gnome.org//extension/602/window-list/"
"https://extensions.gnome.org//extension/6/applications-menu/"
)

# Расширения для установки
for extension in "${extensions[@]}"; do
  # Проверка наличия расширения на сайте extensions.gnome.org
  if ! curl --head --silent --fail "$extension" >/dev/null; then
    echo "Расширение не найдено на сайте extensions.gnome.org."
    continue
  fi

  # Установка расширения
  gnome-extensions install --force "$extension"
  echo "Расширение с URL '$extension' успешно установлено."
done
