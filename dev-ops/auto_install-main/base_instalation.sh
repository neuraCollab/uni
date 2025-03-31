#! /bin/bash

# import modules
. "../general_scripts/check_install.sh"
. "../general_scripts/install_script.sh"
. "../general_scripts/read_env.sh"

# middleware
. "./middleware/pre_instalation.sh"
. "./middleware/post.instalation.sh"

# General constants
export current_programm_name=""

# Массив с вариантами меню
options=("Вариант 1" "Вариант 2" "Вариант 3" "Выйти")

# Функция для вывода меню
show_menu() {
    echo "Мультиселектор меню"
    for ((i = 0; i < ${#options[@]}; i++)); do
        echo "[$((i + 1))] ${options[i]}"
    done
    echo "Введите номер выбранного варианта (через запятую) или q для выхода:"
}

# Функция для обработки выбранных пунктов меню
process_selection() {
    selected_options=()
    input=${1//,/ } # Заменяем запятые на пробелы

    # Разделяем строку на отдельные значения
    IFS=' ' read -ra selection <<<"$input"

    # Проверяем каждое значение и сохраняем выбранные варианты в массив
    for choice in "${selection[@]}"; do
        if [[ "$choice" =~ ^[1-9][0-9]*$ ]] && [ "$choice" -le "${#options[@]}" ]; then
            selected_options+=("${options[choice - 1]}")
        fi
    done

    # Выводим выбранные варианты
    echo "Выбрано:"
    for option in "${selected_options[@]}"; do
        echo "$option"
        # вызываем установку файлов
        # 
        # выполняем действия после
        # . post.instalation.sh "current_programm_name"
    done
    echo "------------"
}

# Главный цикл
while true; do
    show_menu
    read choice

    if [[ "$choice" == "q" ]]; then
        echo "Выход..."
        break
    fi

    process_selection "$choice"
done
