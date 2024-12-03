from functools import reduce

students = [
    {"name": "Alice", "age": 20, "grades": [85, 90, 88, 92]},
    {"name": "Bob", "age": 22, "grades": [78, 89, 76, 85]},
    {"name": "Charlie", "age": 21, "grades": [92, 95, 88, 94]},
    {"name": "David", "age": 23, "grades": [80, 85, 78, 88]},
    {"name": "Eve", "age": 20, "grades": [90, 87, 93, 89]},
    {"name": "Frank", "age": 22, "grades": [75, 80, 85, 79]},
    {"name": "Grace", "age": 21, "grades": [91, 88, 85, 92]},
    {"name": "Hannah", "age": 20, "grades": [82, 77, 89, 84]},
    {"name": "Ian", "age": 22, "grades": [84, 90, 85, 87]},
    {"name": "Jack", "age": 23, "grades": [79, 82, 75, 80]},
    {"name": "Karen", "age": 21, "grades": [90, 91, 89, 92]},
    {"name": "Leo", "age": 20, "grades": [86, 80, 82, 84]},
    {"name": "Megan", "age": 22, "grades": [91, 92, 88, 89]},
    {"name": "Nathan", "age": 21, "grades": [78, 84, 80, 82]},
    {"name": "Olivia", "age": 23, "grades": [95, 93, 94, 96]},
    {"name": "Paul", "age": 20, "grades": [83, 80, 82, 85]},
    {"name": "Quincy", "age": 22, "grades": [79, 85, 87, 81]},
    {"name": "Rachel", "age": 21, "grades": [92, 94, 90, 91]},
    {"name": "Sam", "age": 23, "grades": [77, 80, 76, 81]},
    {"name": "Tina", "age": 20, "grades": [88, 90, 85, 87]},
    {"name": "Umar", "age": 22, "grades": [82, 79, 84, 86]},
    {"name": "Vera", "age": 21, "grades": [91, 88, 87, 89]},
    {"name": "Wendy", "age": 23, "grades": [85, 83, 82, 80]},
    {"name": "Xavier", "age": 20, "grades": [80, 85, 78, 84]},
    {"name": "Yara", "age": 22, "grades": [88, 91, 90, 87]},
    {"name": "Zach", "age": 21, "grades": [79, 80, 83, 85]},
    {"name": "Aiden", "age": 23, "grades": [90, 89, 85, 92]},
    {"name": "Bella", "age": 20, "grades": [83, 87, 85, 88]},
    {"name": "Carter", "age": 22, "grades": [78, 82, 80, 85]},
    {"name": "Daisy", "age": 21, "grades": [92, 91, 90, 93]}
]

# Функция для вычисления среднего балла
def calculate_average(grades):
    return sum(grades) / len(grades)

# Фильтрация студентов по возрасту с использованием filter
def filter_students_by_age(students, age=None):
    return list(filter(lambda student: student['age'] == age if age is not None else True, students))

# Вычисление среднего балла для каждого студента с использованием map
def get_student_averages(students):
    return list(map(lambda student: {**student, "average": calculate_average(student["grades"])}, students))

# Вычисление общего среднего балла по всем студентам с использованием reduce
def overall_average(students):
    total_grades = reduce(lambda acc, student: acc + sum(student["grades"]), students, 0)
    total_subjects = reduce(lambda acc, student: acc + len(student["grades"]), students, 0)
    return total_grades / total_subjects

# Нахождение студентов с самым высоким средним баллом с использованием filter и max
def top_students(students):
    max_average = max(map(lambda student: student["average"], students))
    return list(filter(lambda student: student["average"] == max_average, students))

# Запрос возраста у пользователя
age_input = input("Введите возраст для фильтрации студентов: ")

# Проверка, если введенное значение является числом, преобразуем в int
if age_input.isdigit():
    age = int(age_input)
else:
    print("Некорректный ввод, будет выведен список всех студентов.")
    age = None

# Фильтрация студентов по возрасту
filtered_students = filter_students_by_age(students, age=age)
students_with_averages = get_student_averages(filtered_students)
overall_avg = overall_average(filtered_students)
top_student_list = top_students(students_with_averages)

# Форматированный вывод результатов
print("\nСредний балл каждого студента:")
for student in students_with_averages:
    print(f"Имя: {student['name']}, Возраст: {student['age']}, Средний балл: {student['average']:.2f}")

print("\nОбщий средний балл по всем студентам: {:.2f}".format(overall_avg))

print("\nЛучший студент(ы):")
for student in top_student_list:
    print(f"Имя: {student['name']}, Возраст: {student['age']}, Средний балл: {student['average']:.2f}")
