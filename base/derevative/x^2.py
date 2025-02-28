import numpy as np
import matplotlib.pyplot as plt

# Функция и ее производная
def f(x):
    return x**2

def f_prime(x):
    return 2 * x

# Точка, в которой строим касательную
x0 = 2
y0 = f(x0)
slope = f_prime(x0)  # Производная в точке x0

# Уравнение касательной: y = slope * (x - x0) + y0
def tangent_line(x):
    return slope * (x - x0) + y0

# Диапазон для графика
x = np.linspace(0, 4, 500)
y = f(x)

# Дифференциал
dx = 0.5  # Изменение аргумента
dy = slope * dx  # Приблизительное изменение функции

# Точка для визуализации дифференциала
x1 = x0 + dx
y1 = y0 + dy

# Построение графика
plt.figure(figsize=(8, 6))

# График функции
plt.plot(x, y, label="$f(x) = x^2$", color="blue")

# Точка касания
plt.scatter([x0], [y0], color="red", label="Точка касания $(x_0, f(x_0))$")

# Касательная
x_tangent = np.linspace(1.5, 2.5, 100)
plt.plot(x_tangent, tangent_line(x_tangent), label="Касательная", color="orange", linestyle="--")

# Дифференциал
plt.arrow(x0, y0, dx, dy, color="green", head_width=0.1, length_includes_head=True, label="Дифференциал $df$")
plt.scatter([x1], [y1], color="green", label="$(x_0 + dx, f(x_0) + df)$")

# Аннотации для предела и производной
plt.annotate("$f'(x_0) = \\frac{\\Delta f}{\\Delta x}$",
             xy=(2.2, 4.5), xytext=(2.8, 7),
             arrowprops=dict(facecolor='black', arrowstyle="->"),
             fontsize=12, color="purple")

# Настройки графика
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
plt.title("График функции, касательной и дифференциала", fontsize=14)
plt.xlabel("$x$", fontsize=12)
plt.ylabel("$y$", fontsize=12)
plt.legend()
plt.grid()

plt.show()
