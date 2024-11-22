from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import time
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

# Глобальные параметры
parameters = {
    "current_speed_coeff": 0.5,
    "personal_best_coeff": 1.5,
    "global_best_coeff": 1.5,
    "num_particles": 30,
    "num_iterations": 1,
    "executed_iterations": 0,
    "velocity_limit": 20,
    "inertia_start": 0.9,
    "inertia_end": 0.4,
    "use_inertia": True  # Включение/выключение инерции
}
particles = []
global_best = {
    "position": None,
    "value": float('inf')
}

# Глобальные параметры интерфейса
fig = Figure(figsize=(5, 4), dpi=120)
ax = fig.add_subplot(111)
canvas = None
best_solution_label = None
function_value_label = None
num_iterations_label = None
execution_time_label = None
num_current_iterations_label = None

# Целевая функция
def evaluate_fitness(position):
    x, y = position
    return 4 * (x - 5) ** 2 + (y - 6) ** 2

# Инициализация частицы
def create_particle():
    position = np.array([random.uniform(-500, 500), random.uniform(-500, 500)])
    velocity = np.array([random.uniform(-10, 10), random.uniform(-10, 10)])
    fitness = evaluate_fitness(position)
    return {
        "position": position,
        "velocity": velocity,
        "personal_best_position": position.copy(),
        "personal_best_value": fitness
    }

# Обновление скорости частицы
def update_velocity(particle, inertia_weight):
    global parameters, global_best
    velocity = particle["velocity"]
    position = particle["position"]
    personal_best = particle["personal_best_position"]

    inertia = (inertia_weight * velocity) if parameters["use_inertia"] else velocity
    cognitive = parameters["personal_best_coeff"] * random.random() * (personal_best - position)

    if global_best["position"] is not None:
        social = parameters["global_best_coeff"] * random.random() * (global_best["position"] - position)
    else:
        social = np.array([0.0, 0.0])

    new_velocity = inertia + cognitive + social
    speed = np.linalg.norm(new_velocity)
    if speed > parameters["velocity_limit"]:
        new_velocity = (new_velocity / speed) * parameters["velocity_limit"]
    particle["velocity"] = new_velocity

# Обновление позиции частицы
def update_position(particle):
    particle["position"] += particle["velocity"]
    particle["position"] = np.clip(particle["position"], -500, 500)

# Обновление персонального лучшего значения частицы
def update_personal_best(particle):
    fitness = evaluate_fitness(particle["position"])
    if fitness < particle["personal_best_value"]:
        particle["personal_best_position"] = particle["position"].copy()
        particle["personal_best_value"] = fitness

# Инициализация всех частиц
def initialize_particles():
    global particles, global_best, parameters
    parameters["executed_iterations"] = 0
    particles = [create_particle() for _ in range(parameters["num_particles"])]
    global_best = {"position": None, "value": float('inf')}
    draw_swarm_plot()
    update_iterations_label()

# Обновление метки текущих итераций
def update_iterations_label():
    global num_current_iterations_label
    num_current_iterations_label.config(text=f"Количество выполненных итераций: {parameters['executed_iterations']}")

# Запуск одной итерации
def run_iteration():
    global particles, global_best, parameters
    inertia_weight = max(
        parameters["inertia_start"] - parameters["executed_iterations"] * 
        (parameters["inertia_start"] - parameters["inertia_end"]) / parameters["num_iterations"],
        parameters["inertia_end"]
    )

    for particle in particles:
        update_velocity(particle, inertia_weight)
        update_position(particle)
        update_personal_best(particle)

        if particle["personal_best_value"] < global_best["value"]:
            global_best["position"] = particle["personal_best_position"].copy()
            global_best["value"] = particle["personal_best_value"]

    parameters["executed_iterations"] += 1
    update_iterations_label()
    draw_swarm_plot()
    display_best_solution()

# Отрисовка графика
def draw_swarm_plot():
    global canvas, ax
    ax.clear()
    ax.set_title("Решения", fontsize=10)
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Y", fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=6)

    if particles:
        positions = np.array([p["position"] for p in particles])
        ax.plot(positions[:, 0], positions[:, 1], 'bo', markersize=2)

    if global_best["position"] is not None:
        ax.plot(global_best["position"][0], global_best["position"][1], 'ro', markersize=5)

    if canvas:
        canvas.draw()
    else:
        canvas = FigureCanvasTkAgg(fig, master=frame_plot)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0)

# Запуск нескольких итераций
def run_iterations(num_iterations):
    if not particles:
        messagebox.showerror("Ошибка", "Частицы не созданы. Пожалуйста, инициализируйте частицы перед запуском.")
        return

    start_time = time.time()
    for _ in range(num_iterations):
        run_iteration()
    end_time = time.time()
    execution_time_label.config(text=f"Время выполнения: {end_time - start_time:.2f} секунд")

# Отображение лучшего решения
def display_best_solution():
    global best_solution_label, function_value_label, num_iterations_label, global_best
    if global_best["position"] is not None:
        best_solution_label.config(text=f"Лучшее решение: X[1] = {global_best['position'][0]} \nX[2] = {global_best['position'][1]}")
        function_value_label.config(text=f"Значение функции: {global_best['value']}")
        num_iterations_label.config(text=f'Количество итераций: {parameters["executed_iterations"]}')

# Обновление параметров
def update_global_variable(entry, key):
    try:
        value = float(entry.get())
        parameters[key] = value
    except ValueError:
        print("Некорректное значение")

# Переключение инерции
def toggle_inertia():
    parameters["use_inertia"] = not parameters["use_inertia"]

# Интерфейс
def create_gui():
    global best_solution_label, function_value_label, num_iterations_label, execution_time_label, frame_plot, num_current_iterations_label
    root = tk.Tk()
    root.title("Поставьте зачет пожалуйста :)")

    frame_params = tk.LabelFrame(root, text="Параметры")
    frame_params.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    tk.Label(frame_params, text="Функция").grid(row=0, column=0, sticky="w")
    tk.Label(frame_params, text="4(x1 - 5)^2 + (x2 - 6)^2", width=30).grid(row=0, column=1)

    tk.Label(frame_params, text="Начальная инерция:").grid(row=1, column=0, sticky="w")
    inertia_start_entry = tk.Entry(frame_params, width=5)
    inertia_start_entry.grid(row=1, column=1)
    inertia_start_entry.insert(0, parameters["inertia_start"])
    inertia_start_entry.bind("<Leave>", lambda e: update_global_variable(inertia_start_entry, "inertia_start"))

    tk.Label(frame_params, text="Конечная инерция:").grid(row=2, column=0, sticky="w")
    inertia_end_entry = tk.Entry(frame_params, width=5)
    inertia_end_entry.grid(row=2, column=1)
    inertia_end_entry.insert(0, parameters["inertia_end"])
    inertia_end_entry.bind("<Leave>", lambda e: update_global_variable(inertia_end_entry, "inertia_end"))

    tk.Label(frame_params, text="Количество частиц:").grid(row=3, column=0, sticky="w")
    num_particles_entry = tk.Entry(frame_params, width=5)
    num_particles_entry.grid(row=3, column=1)
    num_particles_entry.insert(0, parameters["num_particles"])
    num_particles_entry.bind("<Leave>", lambda e: update_global_variable(num_particles_entry, "num_particles"))

    tk.Checkbutton(frame_params, text="Использовать инерцию", variable=tk.BooleanVar(value=True), command=toggle_inertia).grid(row=4, column=0, sticky="w")

    frame_control = tk.LabelFrame(root, text="Управление")
    frame_control.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

    tk.Button(frame_control, text="Создать частицы", command=initialize_particles).grid(row=0, column=0, pady=5)

    tk.Label(frame_control, text="Запустить итерации:").grid(row=1, column=0, sticky="w")
    tk.Button(frame_control, text="1", command=lambda: run_iterations(1)).grid(row=1, column=1)
    tk.Button(frame_control, text="10", command=lambda: run_iterations(10)).grid(row=1, column=2)
    tk.Button(frame_control, text="100", command=lambda: run_iterations(100)).grid(row=1, column=3)
    tk.Button(frame_control, text="1000", command=lambda: run_iterations(1000)).grid(row=1, column=4)

    num_current_iterations_label = tk.Label(frame_control, text="Количество выполненных итераций: 0")
    num_current_iterations_label.grid(row=2, column=0, sticky="w")

    frame_plot = tk.LabelFrame(root, text="График")
    frame_plot.grid(row=0, column=1, rowspan=3, padx=10, pady=10, sticky="nsew")

    frame_results = tk.LabelFrame(root, text="Результаты")
    frame_results.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

    best_solution_label = tk.Label(frame_results, text="Лучшее решение:")
    best_solution_label.grid(row=0, column=0, sticky="w")
    num_iterations_label = tk.Label(frame_results, text="Количество итераций:")
    num_iterations_label.grid(row=1, column=0, sticky="w")
    function_value_label = tk.Label(frame_results, text="Значение функции:")
    function_value_label.grid(row=2, column=0, sticky="w")
    execution_time_label = tk.Label(frame_results, text="Время выполнения:")
    execution_time_label.grid(row=3, column=0, sticky="w")

    draw_swarm_plot()
    root.mainloop()

if __name__ == "__main__":
    create_gui()
