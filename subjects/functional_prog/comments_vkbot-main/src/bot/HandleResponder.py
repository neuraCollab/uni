import csv
import re
import os
import random
import pandas as pd



def find_and_select_random_line_v2(file_path, text):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path, on_bad_lines='skip')

        # Check if the first column contains the text

        if text not in df.iloc[:, 0].values:
            return "Text not found in the first column."

        # Find the index of the row where the text is found
        start_index = df.index[df.iloc[:, 0] == text].tolist()[0]

        # Find the next row where the first column has text, to define the group
        next_index = df.index[df.iloc[:, 0].notna() & (df.index > start_index)].min()

        # If there's no next index, select until the end of the DataFrame
        if pd.isna(next_index):
            next_index = len(df)

        # Select the rows in the second column that belong to the group
        group_rows = df.iloc[start_index:next_index, 1].dropna().tolist()

        # If there are no rows in the group, return a message
        if not group_rows:
            return "No corresponding rows in the second column for the group."

        # Randomly select one of the rows from the group
        return random.choice(group_rows)

    except pd.errors.ParserError as e:
        return f"Error parsing CSV: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

        
# print(find_and_select_random_line_v2('../csv/хранитель.csv', "хранитель"))

def process_text(text):
    # Ищем фразу в кавычках с помощью регулярного выражения
    matches = re.findall(r'"(.*?)"', text)

    # Если найдены фразы в кавычках
    for match in matches:
        # Убираем пробелы из начала и конца фразы
        processed_phrase = match.strip()

        # Заменяем исходную фразу в тексте на обработанную
        text = text.replace(f'"{match}"', f'"{processed_phrase}"')

    return text


def find_file_in_directory(directory, filename):
    # Получаем список файлов в заданном каталоге
    files = os.listdir(directory)

    # Проверяем наличие файла с заданным именем
    for file in files:
        if file == filename:
            # Формируем полный путь к файлу
            file_path = os.path.join(directory, filename)

            # Проверяем, является ли найденный путь файлом
            if os.path.isfile(file_path):
                return file_path

    # Если файл не найден, возвращаем None
    return None
