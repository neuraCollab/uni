import numpy as np

# Целевая функция
def f(x):
    x1, x2 = x
    return 100 * (x2 - x1**2)**2 + (1 - x1)**2

# Градиент
def grad_f(x):
    x1, x2 = x
    df_dx1 = -400 * x1 * (x2 - x1**2) - 2 * (1 - x1)
    df_dx2 = 200 * (x2 - x1**2)
    return np.array([df_dx1, df_dx2])

# Матрица Гессе
def hess_f(x):
    x1, x2 = x
    H = np.array([
        [-400 * (x2 - 3 * x1**2) + 2, -400 * x1],
        [-400 * x1, 200]
    ])
    return H

# 1. Метод градиентного спуска
def gradient_descent(f, grad_f, x0, max_iter=1000, tol=1e-6, alpha=0.001):
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    
    for i in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break
        x = x - alpha * g
        history.append(x.copy())
    
    return x, history

# 2. Метод Ньютона
def newton_method(f, grad_f, hess_f, x0, max_iter=100, tol=1e-6):
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    
    for i in range(max_iter):
        g = grad_f(x)
        H = hess_f(x)
        
        if np.linalg.norm(g) < tol:
            break
        
        try:
            p = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            print("Матрица Гессе вырождена")
            break
        
        x = x + p
        history.append(x.copy())
    
    return x, history

# 3. Метод сопряжённых градиентов (Fletcher–Reeves)
def conjugate_gradient(f, grad_f, x0, max_iter=1000, tol=1e-6):
    x = np.array(x0, dtype=float)
    g = grad_f(x)
    d = -g
    history = [x.copy()]
    
    for i in range(max_iter):
        # Одномерная оптимизация (линейный поиск с фиксированным шагом)
        def phi(alpha):
            return f(x + alpha * d)
        
        # Простой подбор шага (можно заменить на более точный)
        alpha = 0.001
        
        x_new = x + alpha * d
        g_new = grad_f(x_new)
        
        if np.linalg.norm(g_new) < tol:
            break
        
        beta = (g_new @ g_new) / (g @ g)
        d = -g_new + beta * d
        
        x = x_new
        g = g_new
        history.append(x.copy())
    
    return x, history

# 4. Квазиньютоновский метод BFGS (уже реализован в предыдущем ответе)
def quasi_newton_bfgs(f, grad_f, x0, max_iter=1000, tol=1e-6, max_step=10.0):
    x = np.array(x0, dtype=float)
    H = np.eye(len(x))
    history = [x.copy()]

    for i in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break

        p = -H @ g

        # Ограничиваем длину шага
        if np.linalg.norm(p) > max_step:
            p = p / np.linalg.norm(p) * max_step

        # Линейный поиск с условием Армихо
        alpha = 1.0
        while alpha > 1e-10:
            try:
                x_new = x + alpha * p
                if np.isnan(x_new).any() or np.isinf(x_new).any():
                    raise ValueError("Новое значение содержит NaN или Inf")
                g_new = grad_f(x_new)
                if f(x_new) <= f(x) + 1e-4 * alpha * g @ p:
                    break
                else:
                    alpha *= 0.5
            except:
                alpha *= 0.5

        if alpha <= 1e-10:
            print("Шаг стал слишком маленьким")
            break

        y = g_new - g
        s = x_new - x
        rho = 1.0 / (y @ s + 1e-10)

        Hy = H @ y
        term1 = np.outer(s, s)
        term2 = np.outer(Hy, s)
        term3 = np.outer(s, Hy)
        term4 = rho * (Hy @ y) * H

        H = H + rho * (term1 - term2 - term3 + term4)

        x = x_new
        history.append(x.copy())

    return x, history

def nelder_mead(f, x0, max_iter=1000, tol=1e-6, alpha=1.0, beta=0.5, gamma=2.0):
    n = len(x0)
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        vertex = np.array(x0, dtype=float)
        vertex[i] += 0.1
        simplex.append(vertex)

    history = [np.mean(simplex, axis=0)]

    for iteration in range(max_iter):
        # Сортировка
        simplex_sorted = sorted(simplex, key=lambda point: f(point))
        x_best = simplex_sorted[0]
        x_second_worst = simplex_sorted[-2]
        x_worst = simplex_sorted[-1]

        # Центр тяжести без худшей точки
        x_centroid = np.mean(simplex_sorted[:-1], axis=0)

        # Отражение
        x_reflected = x_centroid + alpha * (x_centroid - x_worst)

        # Оценка
        f_reflected = f(x_reflected)
        f_best = f(x_best)
        f_second_worst = f(x_second_worst)
        f_worst = f(x_worst)

        # Растяжение
        if f_reflected < f_best:
            x_expanded = x_centroid + gamma * (x_reflected - x_centroid)
            if f(x_expanded) < f_reflected:
                x_new = x_expanded
            else:
                x_new = x_reflected
        # Сжатие
        elif f_reflected >= f_second_worst:
            x_contracted = x_centroid + beta * (x_worst - x_centroid)
            if f(x_contracted) < f_worst:
                x_new = x_contracted
            else:
                # Уменьшение размера
                x_new_points = [x_best + 0.5 * (x - x_best) for x in simplex]
                simplex = x_new_points
                continue
        else:
            x_new = x_reflected

        # Удаление x_worst из simplex
        for idx, point in enumerate(simplex):
            if np.allclose(point, x_worst):
                del simplex[idx]
                break

        simplex.append(x_new)
        history.append(np.mean(simplex, axis=0))

        # Критерий остановки
        centroid = np.mean(simplex, axis=0)
        std_dev = np.std([np.linalg.norm(x - centroid) for x in simplex])
        if std_dev < tol:
            break

    best_x = min(simplex, key=lambda x: f(x))
    return best_x, history

def gradient_descent_adaptive(f, grad_f, x0, max_iter=1000, tol=1e-6, c1=1e-4, alpha_init=1.0):
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    
    for i in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break
        
        # Направление спуска
        p = -g
        
        # Линейный поиск (с условием Армихо)
        alpha = alpha_init
        while alpha > 1e-10:
            x_new = x + alpha * p
            if f(x_new) <= f(x) + c1 * alpha * g @ p:
                break
            else:
                alpha *= 0.5
        else:
            print("Шаг стал слишком маленьким")
            break
        
        x = x_new
        history.append(x.copy())
    
    return x, history
# === Тестирование ===
x0 = [2.0, 2.0]

print("=== Метод градиентного спуска ===")
x_gd, hist_gd = gradient_descent(f, grad_f, x0)
print("x_min =", x_gd)
print("f(x_min) =", f(x_gd))

print("\n=== Метод Ньютона ===")
x_nt, hist_nt = newton_method(f, grad_f, hess_f, x0)
print("x_min =", x_nt)
print("f(x_min) =", f(x_nt))

print("\n=== Метод сопряжённых градиентов ===")
x_cg, hist_cg = conjugate_gradient(f, grad_f, x0)
print("x_min =", x_cg)
print("f(x_min) =", f(x_cg))

# === Метод градиентного спуска (с адаптивным шагом) ===
print("\n=== Градиентный спуск (адаптивный шаг) ===")
x_gd_adaptive, hist_gd_adaptive = gradient_descent_adaptive(f, grad_f, x0)
print("x_min =", x_gd_adaptive)
print("f(x_min) =", f(x_gd_adaptive))

# === Квазиньютоновский метод (BFGS) ===
print("\n=== Квазиньютоновский метод (BFGS) ===")
x_bfgs, hist_bfgs = quasi_newton_bfgs(f, grad_f, x0)
print("x_min =", x_bfgs)
print("f(x_min) =", f(x_bfgs))

print("\n=== Метод Нелдера — Мида ===")
x_nm, hist_nm = nelder_mead(f, x0)
print("x_min =", x_nm)
print("f(x_min) =", f(x_nm))