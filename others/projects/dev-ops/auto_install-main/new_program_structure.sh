#!/bin/bash

if [ $# -lt 1 ]; then
  echo "Usage: $0 <directory_name> <distribution_name>"
  exit 1
fi

directory_name=$1
distribution_name=$2

if [ -z "$distribution_name" ]; then
  distribution_name="ubuntu"  # Присвоение значения "ubuntu" по умолчанию, если аргумент не был задан
fi

# Создание директории
mkdir "$directory_name"

# Создание первого файла внутри директории
first_file="${directory_name}_installation.md"
touch "$directory_name/$first_file"

# Добавление текста в начало первого файла
echo "# ${directory_name} instalation" > "$directory_name/$first_file"

# Создание второго файла внутри директории
second_file="${directory_name}_${distribution_name}.sh"
touch "$directory_name/$second_file"

# Добавление текста в начало второго файла
current_program_name="$directory_name"
echo "#!/bin/bash\n\ncurrent_program_name=\"$current_program_name\"" > "$current_program_name/$second_file"

echo "Директория и файлы успешно созданы и изменены."