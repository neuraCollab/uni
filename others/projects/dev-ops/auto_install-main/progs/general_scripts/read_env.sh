# read_env_file

read_env_file() {
  file_path="../.env"
  
  # Проверка существования файла
  if [ ! -f "$file_path" ]; then
    echo "Файл $file_path не найден."
    return 1
  fi
  
  # Чтение файла и экспорт переменных среды
  while IFS= read -r line; do
    # Игнорирование комментариев и пустых строк
    if [[ "$line" =~ ^[[:space:]]*#|^$ ]]; then
      continue
    fi
    
    # Разделение строки на имя переменной и значение
    variable=$(echo "$line" | cut -d '=' -f1)
    value=$(echo "$line" | cut -d '=' -f2-)
    
    # Удаление возможных кавычек в значении переменной
    value="${value%\"}"
    value="${value#\"}"
    
    # Экспорт переменной среды
    export "$variable"="$value"
    
    echo "Прочитана переменная среды: $variable=$value"
  done < "$file_path"
  
  echo "Переменные среды из файла $file_path успешно загружены."
  return 0
}