#! /bin/bash

# check_install.sh

# Return 1 if installed and 0 if not
check_install() {
  # Имя приложения для проверки
  program_name="$1"

  # Проверяем установку приложения
  if command -v "$program_name" &>/dev/null; then
    echo "Программа $program_name уже установлена."
    return 1
  else
    echo "Программа $program_name не установлена."
    return 0
  fi

}
