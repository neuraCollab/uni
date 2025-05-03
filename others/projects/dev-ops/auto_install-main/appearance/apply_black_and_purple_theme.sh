#!/bin/bash

# Проверка наличия установленной утилиты gsettings
if ! command -v gsettings &>/dev/null; then
  echo "Утилита 'gsettings' не найдена. Убедитесь, что вы используете среду рабочего стола GNOME."
  exit 1
fi

# Задание черной темы оформления
gsettings set org.gnome.desktop.interface gtk-theme "Adwaita-dark"
gsettings set org.gnome.desktop.interface cursor-theme "Yaru"
gsettings set org.gnome.desktop.interface icon-theme "Yaru"
gsettings set org.gnome.desktop.interface document-font-name "Ubuntu 11"
gsettings set org.gnome.desktop.interface monospace-font-name "Ubuntu Mono 13"
gsettings set org.gnome.desktop.wm.preferences titlebar-font "Ubuntu Bold 11"

# Задание фиолетового цвета обоев рабочего стола
gsettings set org.gnome.desktop.background primary-color "#8a2be2"

echo "Черная тема оформления и фиолетовый цвет успешно применены."
