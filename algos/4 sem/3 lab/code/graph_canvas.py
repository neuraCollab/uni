import tkinter as tk
import math
import random
import json
from tkinter import messagebox
from utils import point_to_segment_dist
from graph_config import GraphConfig
import time

class GraphCanvas(tk.Canvas):
    def __init__(self, master=None, config=None, **kwargs):
        super().__init__(master, **kwargs)
        self.config = config if config else GraphConfig()
        self.vertices = []
        self.edges = []
        self.add_edge_mode = False
        self.first_vertex_index = None
        self.var_directed = None
        self.var_weighted = None
        self.entry_weight = None
        self.bind("<Button-1>", self.on_left_click)
        self.bind("<Button-3>", self.on_right_click)

    def solve_tsp_simulated_annealing(self, initial_temp=1000, cooling_rate=0.995, cauchy=False):
        """Решение задачи коммивояжера методом имитации отжига"""
        n = len(self.vertices)
        if n < 2:
            return None, float('inf'), 0

        # Строим матрицу расстояний
        dist = [[math.inf]*n for _ in range(n)]
        for i, j, d in self.edges:
            dist[i][j] = d

        start_time = time.time()
        current_path = list(range(n))
        random.shuffle(current_path)
        best_path = current_path[:]

        def tour_length(path):
            length = 0
            for k in range(n):
                i = path[k]
                j = path[(k+1)%n]
                if dist[i][j] == math.inf:
                    return math.inf
                length += dist[i][j]
            return length

        current_length = tour_length(current_path)
        best_length = current_length
        temp = initial_temp
        iteration = 1

        while temp > 1e-3:
            # Генерируем соседнее решение
            i, j = random.sample(range(n), 2)
            neighbor = current_path[:]
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbor_length = tour_length(neighbor)
            
            # Критерий принятия решения
            delta = neighbor_length - current_length
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_path = neighbor
                current_length = neighbor_length
                if current_length < best_length:
                    best_path = current_path[:]
                    best_length = current_length

            # Обновление температуры
            if cauchy:
                temp = initial_temp / (1 + cooling_rate * iteration)
            else:
                temp *= cooling_rate
            iteration += 1

        end_time = time.time()
        elapsed = end_time - start_time
        
        # Замыкаем цикл
        best_path.append(best_path[0])
        return best_path, best_length, elapsed

    def solve_tsp_aco(self, num_ants=20, num_iters=100, alpha=1.0, beta=5.0, rho=0.5, wandering=True):
        """Решение задачи коммивояжера методом муравьиной колонии
        
        Args:
            num_ants (int): Количество муравьев
            num_iters (int): Количество итераций
            alpha (float): Вес феромона
            beta (float): Вес эвристики
            rho (float): Коэффициент испарения феромона
            wandering (bool): Использовать блуждающую колонию (True) или фиксированное начало (False)
        """
        n = len(self.vertices)
        if n < 2:
            return None, float('inf'), 0

        # Матрица расстояний
        dist = [[math.inf]*n for _ in range(n)]
        for i, j, d in self.edges:
            dist[i][j] = d

        # Инициализация феромона
        tau0 = 1.0 / (n * sum(min(row) for row in dist if min(row) < math.inf))
        tau = [[tau0]*n for _ in range(n)]
        
        # Эвристика η = 1/d
        eta = [[0 if dist[i][j]==math.inf else 1.0/dist[i][j] for j in range(n)] for i in range(n)]

        best_path, best_len = None, float('inf')
        start_time = time.time()

        for iteration in range(num_iters):
            all_paths = []
            
            # Запуск муравьев
            for ant in range(num_ants):
                # Выбор начальной вершины
                if wandering:
                    # Блуждающая колония: случайная стартовая вершина для каждого муравья
                    current = random.randrange(n)
                else:
                    # Фиксированное начало: все муравьи стартуют из вершины 0
                    current = 0
                    
                visited = {current}
                path = [current]

                # Построение пути
                while len(path) < n:
                    i = path[-1]
                    # Вероятности перехода
                    probs = []
                    total = 0
                    
                    # Расчет вероятностей для всех доступных вершин
                    for j in range(n):
                        if j not in visited and dist[i][j] < math.inf:
                            prob = (tau[i][j]**alpha) * (eta[i][j]**beta)
                            probs.append((j, prob))
                            total += prob
                    
                    if not probs:
                        break
                        
                    # Выбор следующей вершины
                    r = random.random() * total
                    cum_prob = 0
                    for j, prob in probs:
                        cum_prob += prob
                        if cum_prob >= r:
                            next_vertex = j
                            break
                    else:
                        next_vertex = probs[-1][0]
                    
                    visited.add(next_vertex)
                    path.append(next_vertex)

                # Замыкаем путь
                if len(path) == n and dist[path[-1]][path[0]] < math.inf:
                    path.append(path[0])
                    length = sum(dist[path[k]][path[k+1]] for k in range(n))
                    all_paths.append((path, length))

                    if length < best_len:
                        best_len = length
                        best_path = path[:]

            # Испарение феромона
            for i in range(n):
                for j in range(n):
                    tau[i][j] *= (1 - rho)
                    
            # Обновление феромона
            for path, length in all_paths:
                delta = 1.0 / length
                for k in range(len(path)-1):
                    i, j = path[k], path[k+1]
                    tau[i][j] += delta

        elapsed = time.time() - start_time
        return best_path, best_len, elapsed

    def set_add_edge_mode(self, enabled):
        self.add_edge_mode = enabled
        self.first_vertex_index = None

    def clear_graph(self):
        self.vertices.clear()
        self.edges.clear()
        self.redraw()

    def on_left_click(self, event):
        x, y = event.x, event.y
        if self.add_edge_mode:
            idx = self.find_nearest_vertex(x, y)
            if idx is not None:
                if self.first_vertex_index is None:
                    self.first_vertex_index = idx
                else:
                    second_vertex_index = idx
                    if second_vertex_index != self.first_vertex_index:
                        self.add_edge(self.first_vertex_index, second_vertex_index)
                    self.first_vertex_index = None
            self.redraw()
        else:
            name = str(len(self.vertices))
            self.vertices.append({"x": x, "y": y, "name": name})
            self.redraw()

    def on_right_click(self, event):
        x, y = event.x, event.y
        if self.add_edge_mode:
            edge_idx = self.find_nearest_edge(x, y, threshold=5)
            if edge_idx is not None:
                self.edges.pop(edge_idx)
                self.redraw()
        else:
            vertex_idx = self.find_nearest_vertex(x, y, threshold=10)
            if vertex_idx is not None:
                self.remove_vertex(vertex_idx)
                self.redraw()
            else:
                edge_idx = self.find_nearest_edge(x, y, threshold=5)
                if edge_idx is not None:
                    self.edges.pop(edge_idx)
                    self.redraw()

    def find_nearest_vertex(self, x, y, threshold=10):
        for i, v in enumerate(self.vertices):
            dx = v["x"] - x
            dy = v["y"] - y
            dist = math.hypot(dx, dy)
            if dist <= threshold:
                return i
        return None

    def find_nearest_edge(self, x, y, threshold=5):
        min_dist = math.inf
        nearest_idx = None
        for idx, (i, j, w) in enumerate(self.edges):
            v1 = self.vertices[i]
            v2 = self.vertices[j]
            dist = point_to_segment_dist(x, y, v1["x"], v1["y"], v2["x"], v2["y"])
            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx
        if min_dist <= threshold:
            return nearest_idx
        return None

    def remove_vertex(self, idx):
        new_edges = []
        for (i, j, w) in self.edges:
            if i != idx and j != idx:
                new_edges.append((i, j, w))
        self.vertices.pop(idx)
        updated_edges = []
        for (i, j, w) in new_edges:
            new_i = i if i < idx else i - 1
            new_j = j if j < idx else j - 1
            updated_edges.append((new_i, new_j, w))
        self.edges = updated_edges

    def add_edge(self, i, j):
        if not self.edge_exists(i, j):
            weight = self.get_edge_weight(i, j)
            self.edges.append((i, j, weight))
            if self.var_directed and not self.var_directed.get():
                if not self.edge_exists(j, i):
                    self.edges.append((j, i, weight))

    def edge_exists(self, i, j):
        return any(e[0] == i and e[1] == j for e in self.edges)

    def get_edge_weight(self, i, j):
        if self.var_weighted and self.var_weighted.get():
            if self.entry_weight is not None:
                txt = self.entry_weight.get().strip()
                try:
                    return float(txt)
                except:
                    return self.calculate_distance(i, j)
            else:
                return self.calculate_distance(i, j)
        else:
            return self.calculate_distance(i, j)

    def calculate_distance(self, i, j):
        v1 = self.vertices[i]
        v2 = self.vertices[j]
        return math.hypot(v1["x"] - v2["x"], v1["y"] - v2["y"])

    def redraw(self):
        self.delete("all")
        
        # Проверяем и фильтруем рёбра с некорректными индексами
        valid_edges = [(i, j, w) for i, j, w in self.edges 
                      if i < len(self.vertices) and j < len(self.vertices)]
        self.edges = valid_edges
        
        # Сначала рисуем рёбра
        for i, j, w in self.edges:
            self.draw_single_arrow(i, j, w)
        
        # Затем рисуем вершины поверх рёбер
        radius = 10
        for idx, v in enumerate(self.vertices):
            x, y = v["x"], v["y"]
            self.create_oval(
                x - radius, y - radius, x + radius, y + radius,
                fill=self.config.color_vertex,
                outline=self.config.color_vertex_outline,
                width=2
            )
            self.create_text(x + radius + 5, y, text=v["name"], anchor="w")

    def draw_single_arrow(self, i, j, weight):
        v1 = self.vertices[i]
        v2 = self.vertices[j]
        x1, y1 = v1["x"], v1["y"]
        x2, y2 = v2["x"], v2["y"]
        is_directed = (self.var_directed and self.var_directed.get())
        self.create_line(
            x1, y1, x2, y2,
            arrow=(tk.LAST if is_directed else None),
            arrowshape=self.config.arrow_shape,
            width=self.config.edge_width,
            fill=self.config.color_forward
        )
        if self.var_weighted and self.var_weighted.get():
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            self.create_text(
                mid_x, mid_y - 10,
                text=f"{weight:.1f}",
                fill=self.config.color_text_forward
            )

    def draw_double_arrow(self, i, j, w1, w2):
        v1 = self.vertices[i]
        v2 = self.vertices[j]
        x1, y1 = v1["x"], v1["y"]
        x2, y2 = v2["x"], v2["y"]
        is_directed = (self.var_directed and self.var_directed.get())
        self.create_line(
            x1, y1, x2, y2,
            arrow=(tk.LAST if is_directed else None),
            arrowshape=self.config.arrow_shape,
            width=self.config.edge_width,
            fill=self.config.color_forward
        )
        if self.var_weighted and self.var_weighted.get():
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            self.create_text(
                mid_x, mid_y - 10,
                text=f"{w1:.1f}",
                fill=self.config.color_text_forward
            )
        self.create_line(
            x2, y2, x1, y1,
            arrow=(tk.LAST if is_directed else None),
            arrowshape=self.config.arrow_shape,
            width=self.config.edge_width,
            fill=self.config.color_backward
        )
        if self.var_weighted and self.var_weighted.get():
            mid_x2 = (x2 + x1) / 2
            mid_y2 = (y2 + y1) / 2
            self.create_text(
                mid_x2, mid_y2 + 10,
                text=f"{w2:.1f}",
                fill=self.config.color_text_backward
            )

    def load_graph_from_json(self, filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.clear_graph()
            self.vertices = data.get("vertices", [])
            self.edges = [tuple(e) for e in data.get("edges", [])]
            self.redraw()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить граф из файла:\n{e}")

    def generate_random_graph(self, num_vertices=5, edge_prob=0.5, max_weight=10):
        self.clear_graph()
        width = self.winfo_width() or 800
        height = self.winfo_height() or 600
        for i in range(num_vertices):
            x = random.randint(50, width - 50)
            y = random.randint(50, height - 50)
            self.vertices.append({"x": x, "y": y, "name": str(i)})
        for i in range(num_vertices):
            for j in range(num_vertices):
                if i != j and random.random() < edge_prob:
                    weight = random.uniform(1.0, max_weight)
                    self.edges.append((i, j, weight))
                    if self.var_directed and not self.var_directed.get():
                        self.edges.append((j, i, weight))
        self.redraw() 