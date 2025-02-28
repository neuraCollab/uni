from functools import reduce

students = [
    {"name": "Alice", "age": 20, "grades": [85, 90, 88, 92]},
    {"name": "Bob", "age": 22, "grades": [78, 89, 76, 85]},
    {"name": "Charlie", "age": 21, "grades": [92, 95, 88, 94]},
    {"name": "David", "age": 23, "grades": [65, 70, 68, 72]},
    {"name": "Eve", "age": 22, "grades": [88, 87, 90, 91]},
    {"name": "Frank", "age": 20, "grades": [82, 80, 84, 86]},
    {"name": "Grace", "age": 21, "grades": [91, 92, 90, 94]},
    {"name": "Hannah", "age": 23, "grades": [77, 75, 80, 78]},
    {"name": "Isaac", "age": 22, "grades": [69, 73, 71, 75]},
    {"name": "Jack", "age": 21, "grades": [85, 87, 89, 90]},
    {"name": "Karen", "age": 23, "grades": [79, 81, 80, 83]},
    {"name": "Leo", "age": 22, "grades": [92, 94, 90, 93]},
    {"name": "Mia", "age": 21, "grades": [87, 89, 90, 91]},
    {"name": "Nick", "age": 20, "grades": [76, 79, 77, 80]},
    {"name": "Olivia", "age": 23, "grades": [83, 85, 88, 87]},
    {"name": "Paul", "age": 22, "grades": [90, 88, 85, 89]},
    {"name": "Quincy", "age": 21, "grades": [82, 83, 81, 85]},
    {"name": "Rachel", "age": 23, "grades": [88, 89, 90, 92]},
    {"name": "Steve", "age": 22, "grades": [91, 93, 89, 90]},
    {"name": "Tina", "age": 21, "grades": [74, 75, 73, 78]},
    {"name": "Uma", "age": 23, "grades": [81, 83, 85, 84]},
    {"name": "Victor", "age": 22, "grades": [87, 88, 89, 90]},
    {"name": "Wendy", "age": 21, "grades": [91, 93, 92, 95]},
    {"name": "Xander", "age": 20, "grades": [79, 80, 81, 82]},
    {"name": "Yara", "age": 23, "grades": [88, 90, 89, 91]},
    {"name": "Zach", "age": 21, "grades": [86, 87, 88, 90]},
]

age_filter = 20
filtered_students = list(filter(lambda student: student['age'] == age_filter, students))

# Расчет среднего балла для каждого студента
def calculate_average(grades):
    return sum(grades) / len(grades)

for student in students:
    student['average'] = calculate_average(student['grades'])

# Общий средний балл по всем студентам
overall_average = reduce(lambda acc, student: acc + student['average'], students, 0) / len(students)

# Поиск студента с наивысшим средним баллом
max_average = max(student['average'] for student in students)
top_students = list(filter(lambda student: student['average'] == max_average, students))

print("Фильтрованные студенты:")
for student in filtered_students:
    print(student)

print("Средний балл каждого студента:")
for student in map(lambda student: (student['name'], student['average']), students):
    print(student)

print("Общий средний балл по всем студентам:", overall_average)

print("Студент с самым высоким средним баллом:", top_students)
