#! /bin/bash

current_program_name="apache"

if [ -z "$your_domain" ]; then
  read -p "Введите ваш домен: " your_domain  # Попросить пользователя ввести значение для your_domain
  if [ -z "$your_domain" ]; then
    echo "Домен не был предоставлен. Выполнение скрипта остановлено."
    exit 1  # Остановить выполнение скрипта из-за отсутствия значения your_domain
  fi
fi

echo "Используемый домен: $your_domain"

# Let’s begin by updating the local package index to reflect the latest upstream changes:

sudo apt update

sudo apt install curl

# Then, install the apache2 package:

sudo apt install apache2