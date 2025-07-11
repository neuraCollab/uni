import numpy as np

def f(x):
    x1,  x2 = x
    return 100 * (x2 - x1**2)**2 + (1 - x1)**2
def grad_f(x):
    x1, x2 = x
    df_dx1 = -400*x1*(x2 - x1**2) - 2*(1 - x1)
    df_dx2 = 200*(x2 - x1**2)
    return np.array([df_dx1,  df_dx2])

def hess_f(x):
    x1, x2 = x
    H = np.array([
        [-400 * (x2 - 3 * x1**2) + 2, -400 * x1],
        [-400 * x1, 200]
    ])
    return H

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

def conjugate_gradient(f, grad_f, x0, max_iter=1000, tol=1e-6):
    x = np.array(x0, dtype=float)
    g = grad_f(x)
    d = -g
    history = [x.copy()]
    
    for i in range(max_iter):
        def phi(alpha):
            return f(x + alpha * d)
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

def nelder_mead(f, x0, max_iter=1000, tol=1e-6, alpha=1.0, beta=0.5, gamma=2.0):
    n = len(x0)
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        vertex = np.array(x0, dtype=float)
        vertex[i] += 0.1
        simplex.append(vertex)

    history = [np.mean(simplex, axis=0)]

    for iteration in range(max_iter):
        simplex_sorted = sorted(simplex, key=lambda point: f(point))
        x_best = simplex_sorted[0]
        x_second_worst = simplex_sorted[-2]
        x_worst = simplex_sorted[-1]
        x_centroid = np.mean(simplex_sorted[:-1], axis=0)
        x_reflected = x_centroid + alpha * (x_centroid - x_worst)
        f_reflected = f(x_reflected)
        f_best = f(x_best)
        f_second_worst = f(x_second_worst)
        f_worst = f(x_worst)
        if f_reflected < f_best:
            x_expanded = x_centroid + gamma * (x_reflected - x_centroid)
            if f(x_expanded) < f_reflected:
                x_new = x_expanded
            else:
                x_new = x_reflected
        elif f_reflected >= f_second_worst:
            x_contracted = x_centroid + beta * (x_worst - x_centroid)
            if f(x_contracted) < f_worst:
                x_new = x_contracted
            else:
                x_new_points = [x_best + 0.5 * (x - x_best) for x in simplex]
                simplex = x_new_points
                continue
        else:
            x_new = x_reflected
        for idx, point in enumerate(simplex):
            if np.allclose(point, x_worst):
                del simplex[idx]
                break
        simplex.append(x_new)
        history.append(np.mean(simplex, axis=0))
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

print("\n=== Градиентный спуск (адаптивный шаг) ===")
x_gd_adaptive, hist_gd_adaptive = gradient_descent_adaptive(f, grad_f, x0)
print("x_min =", x_gd_adaptive)
print("f(x_min) =", f(x_gd_adaptive))

print("\n=== Метод Нелдера — Мида ===")
x_nm, hist_nm = nelder_mead(f, x0)
print("x_min =", x_nm)
print("f(x_min) =", f(x_nm))