from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image
import io
import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.figure import Figure
from tkinter import messagebox

# Глобальные переменные
current_speed_coeff = 0.5
personal_best_coeff = 1.5
global_best_coeff = 1.5
num_particles = 30
num_iterations = 1
executed_iterations = 0
frame_plot = None
velocity_limit = 20  # Максимальная скорость
# Глобальные переменные для графика
fig = Figure(figsize=(5, 4), dpi=120)
ax = fig.add_subplot(111)
canvas = None

# Глобальные переменные для роевого алгоритма
particles = []
global_best_position = None
global_best_value = float('inf')
best_solution_label = None
function_value_label = None
num_iterations_label = None
num_current_iterations_label = None

# Функция 
def fitness_function(position):
    x, y = position
    return -12*y + 4*(x**2) + 4*(y**2) - 4*x*y
    

def update_iterations_label():
    global num_current_iterations_label, executed_iterations
    num_current_iterations_label.config(text=f"Количество выполненных итераций: {executed_iterations}")

# Класс для частиц
class Particle:
    def __init__(self):
        self.position = np.array([random.uniform(-500, 500), random.uniform(-500, 500)])
        self.velocity = np.array([random.uniform(-10, 10), random.uniform(-10, 10)])
        self.personal_best_position = self.position.copy()
        self.personal_best_value = fitness_function(self.position)

    def update_velocity(self):
        global global_best_position
        inertia = current_speed_coeff * self.velocity
        cognitive = personal_best_coeff * random.random() * (self.personal_best_position - self.position)

        # Проверка на наличие глобального лучшего положения
        if global_best_position is not None:
            social = global_best_coeff * random.random() * (global_best_position - self.position)
        else:
            social = np.array([0.0, 0.0])  # Если нет глобального лучшего положения, social равен нулю

        # Обновляем скорость частицы
        self.velocity = inertia + cognitive + social
        
        # Ограничиваем скорость
        speed = np.linalg.norm(self.velocity)  # Нормируем скорость
        if speed > velocity_limit:  # Если скорость превышает предел
            self.velocity = (self.velocity / speed) * velocity_limit  # Нормализуем скорость до предела

    def update_position(self):
        self.position += self.velocity
        # Ограничиваем положение частицы в пределах поиска
        self.position = np.clip(self.position, -500, 500)

    def update_personal_best(self):
        fitness = fitness_function(self.position)
        if fitness < self.personal_best_value:
            self.personal_best_position = self.position.copy()
            self.personal_best_value = fitness

# Функция для инициализации частиц
def initialize_particles():
    global particles, global_best_position, global_best_value, executed_iterations
    executed_iterations = 0
    update_iterations_label()
    particles = [Particle() for _ in range(num_particles)]
    global_best_position = None
    global_best_value = float('inf')
    draw_swarm_plot()  # вызов функции для обновления графика

def run_iteration():
    global particles, global_best_position, global_best_value, executed_iterations

    for particle in particles:
        particle.update_velocity()
        particle.update_position()
        particle.update_personal_best()
        
        # Обновляем глобальное лучшее значение
        if particle.personal_best_value < global_best_value:
            global_best_position = particle.personal_best_position.copy()
            global_best_value = particle.personal_best_value

    executed_iterations += 1
    
    # Обновление Label с количеством итераций
    update_iterations_label()
    draw_swarm_plot()

    # Обновление лучшего решения
    best_particle = (global_best_value, global_best_position[0], global_best_position[1])
    display_best_solution(best_particle)  # Вызов функции обновления лучшего решения


# Функция для отрисовки графика
def draw_swarm_plot():
    global canvas, ax
    ax.clear()
    ax.set_title("Решения", fontsize=10)
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Y", fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=6)

    # Отрисовка частиц, только если они существуют
    if particles:
        positions = np.array([particle.position for particle in particles])
        ax.plot(positions[:, 0], positions[:, 1], 'bo', markersize=2)
    
    # Отрисовка глобального лучшего положения
    if global_best_position is not None:
        ax.plot(global_best_position[0], global_best_position[1], 'ro', markersize=5)

    # Если холст уже существует, обновляем его, иначе создаем новый
    if canvas:
        canvas.draw()
    else:
        canvas = FigureCanvasTkAgg(fig, master=frame_plot)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0)


def run_iterations():
    global num_iterations
    print(f'num iter from run_iterations: {num_iterations}')

    # Проверка на наличие частиц
    if not particles:
        messagebox.showerror("Ошибка", "Частицы не созданы. Пожалуйста, инициализируйте частицы перед запуском итераций.")
        return  # Прекратить выполнение функции, если частицы не созданы

    for _ in range(num_iterations):
        run_iteration()
        
def display_best_solution(best_particle):
    global best_solution_label, function_value_label, num_iterations_label,executed_iterations
    func_val, x_best_sol, y_best_sol  = best_particle
    best_solution_label.config(text=f"Лучшее решение: X[1] = {x_best_sol} \nX[2] = {y_best_sol}")
    function_value_label.config(text=f"Значение функции: {func_val}")
    num_iterations_label.config(text=f'Количество поколений: {executed_iterations}')
    
# Функция для обновления глобальных переменных
def update_global_variable(entry, variable_name):
    global current_speed_coeff, personal_best_coeff, global_best_coeff, num_particles, num_iterations,velocity_limit
    try:
        value = float(entry.get())
        if variable_name == "current_speed_coeff":
            current_speed_coeff = value
        elif variable_name == "personal_best_coeff":
            personal_best_coeff = value
        elif variable_name == "global_best_coeff":
            global_best_coeff = value
        elif variable_name == "num_particles":
            num_particles = int(value)
        elif variable_name == "num_iterations":
            num_iterations = int(value)
        elif variable_name == "velocity_limit":
            velocity_limit = int(value)
    except ValueError:
        print("Введите правильное значение")
    
def add_to_num_iterations(value: int):
    global num_iterations
    num_iterations = value
    

def create_gui():
    global best_solution_label, function_value_label, num_current_iterations_label, num_iterations_label, current_speed_coeff, personal_best_coeff, global_best_coeff, num_particles, frame_plot,velocity_limit
    root = tk.Tk()
    root.title("Genetic Algorithm")

    frame_params = tk.LabelFrame(root, text="Параметры")
    frame_params.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    tk.Label(frame_params, text="Функция").grid(row=0, column=0, sticky="w")
    tk.Label(frame_params, text="-12y + 4*(x^2) + 4*(y^2) - 4xy", width=30).grid(row=0, column=1)

    tk.Label(frame_params, text="Коэфф. текущей скорости:").grid(row=1, column=0, sticky="w")
    current_speed_entry = tk.Entry(frame_params, width=5)
    current_speed_entry.grid(row=1, column=1)
    current_speed_entry.insert(0, current_speed_coeff)
    current_speed_entry.bind("<Leave>", lambda e: update_global_variable(current_speed_entry, 'current_speed_coeff'))
    
    tk.Label(frame_params, text="Коэфф. собственного лучшего значения").grid(row=2, column=0, sticky="w")
    personal_best_entry = tk.Entry(frame_params, width=5)
    personal_best_entry.grid(row=2, column=1)
    personal_best_entry.insert(0, personal_best_coeff)
    personal_best_entry.bind("<Leave>", lambda e: update_global_variable(personal_best_entry, 'personal_best_coeff'))

    tk.Label(frame_params, text="Коэфф. глобального лучшего значения").grid(row=3, column=0, sticky="w")
    global_best_entry = tk.Entry(frame_params, width=5)
    global_best_entry.grid(row=3, column=1)
    global_best_entry.insert(0, global_best_coeff)
    global_best_entry.bind("<Leave>", lambda e: update_global_variable(global_best_entry, 'global_best_coeff'))

    tk.Label(frame_params, text="Количество частиц").grid(row=4, column=0, sticky="w")
    num_particles_entry = tk.Entry(frame_params, width=5)
    num_particles_entry.grid(row=4, column=1)
    num_particles_entry.insert(0, num_particles)
    num_particles_entry.bind("<Leave>", lambda e: update_global_variable(num_particles_entry, 'num_particles'))
    
    tk.Label(frame_params, text="Максимальная скорость").grid(row=5, column=0, sticky="w")
    velocity_limit_entry = tk.Entry(frame_params, width=5)
    velocity_limit_entry.grid(row=5, column=1)
    velocity_limit_entry.insert(0, velocity_limit)
    velocity_limit_entry.bind("<Leave>", lambda e: update_global_variable(velocity_limit_entry, 'velocity_limit'))

    frame_control = tk.LabelFrame(root, text="Управление")
    frame_control.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

    tk.Button(frame_control, text="Создать частицы", command=initialize_particles).grid(row=0, column=0, columnspan=1, pady=5)

    tk.Label(frame_control, text="Количество итераций:").grid(row=1, column=0, sticky="w")
    tk.Button(frame_control, text="1", command=lambda: add_to_num_iterations(1)).grid(row=1, column=1)
    tk.Button(frame_control, text="10", command=lambda: add_to_num_iterations(10)).grid(row=1, column=2)
    tk.Button(frame_control, text="100", command=lambda: add_to_num_iterations(100)).grid(row=1, column=3)
    tk.Button(frame_control, text="1000", command=lambda: add_to_num_iterations(1000)).grid(row=1, column=4)

    tk.Button(frame_control, text="Запустить итерации", command=run_iterations).grid(row=3, column=0, columnspan=2, pady=5)
    
    num_current_iterations_label = tk.Label(frame_control, text="Количество выполненных итераций: 0")
    num_current_iterations_label.grid(row=4, column=0, sticky="w")


    frame_plot = tk.LabelFrame(root, text="Роевые решения")
    frame_plot.grid(row=0, column=1, rowspan=3, padx=10, pady=10, sticky="nsew")
    
    # Результаты
    frame_results = tk.LabelFrame(root, text="Результаты")
    frame_results.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

    best_solution_label = tk.Label(frame_results, text="Лучшее решение:")
    best_solution_label.grid(row=0, column=0, sticky="w")

    num_iterations_label = tk.Label(frame_results, text="Количество итераций:")
    num_iterations_label.grid(row=1, column=0, sticky="w")
    
    function_value_label = tk.Label(frame_results, text="Значение функции:")
    function_value_label.grid(row=2, column=0, sticky="w")
    
    # Отображаем пустой график при запуске
    draw_swarm_plot()

    root.mainloop()

if __name__ == '__main__':
    create_gui()
