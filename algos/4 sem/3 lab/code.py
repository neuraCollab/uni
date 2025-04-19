import tkinter as tk
import math
import random
import json
import time
from tkinter import filedialog, messagebox

class GraphConfig:
    def __init__(
        self,
        arrow_shape=(16, 20, 6),
        edge_width=2,
        color_forward="black",
        color_backward="black",
        color_vertex="orange",
        color_vertex_outline="black",
        color_text_forward="blue",
        color_text_backward="red"
    ):
        self.arrow_shape = arrow_shape
        self.edge_width = edge_width
        self.color_forward = color_forward
        self.color_backward = color_backward
        self.color_vertex = color_vertex
        self.color_vertex_outline = color_vertex_outline
        self.color_text_forward = color_text_forward
        self.color_text_backward = color_text_backward

# Utility functions

def point_to_segment_dist(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / float(dx*dx + dy*dy)
    if t < 0:
        return math.hypot(px - x1, py - y1)
    elif t > 1:
        return math.hypot(px - x2, py - y2)
    projx = x1 + t * dx
    projy = y1 + t * dy
    return math.hypot(px - projx, py - projy)

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
        edges_dict = {}
        for (i, j, w) in self.edges:
            edges_dict[(i, j)] = w
        drawn = set()
        for (i, j), w in edges_dict.items():
            if (i, j) in drawn:
                continue
            if (j, i) in edges_dict:
                w2 = edges_dict[(j, i)]
                self.draw_double_arrow(i, j, w, w2)
                drawn.add((i, j))
                drawn.add((j, i))
            else:
                self.draw_single_arrow(i, j, w)
                drawn.add((i, j))
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
    def solve_tsp_simulated_annealing(self, initial_temp=1000, cooling_rate=0.995, cauchy=False):
        n = len(self.vertices)
        if n < 2:
            return None, float('inf'), 0

        # Build distance matrix
        dist = [[math.inf]*n for _ in range(n)]
        for i, j, d in self.edges:
            dist[i][j] = d
        # initial solution: random permutation
        current_solution = list(range(n))
        random.shuffle(current_solution)
        best_solution = current_solution[:]
        def tour_length(sol):
            length = 0
            for k in range(n):
                i = sol[k]
                j = sol[(k+1)%n]
                length += dist[i][j]
            return length
        current_length = tour_length(current_solution)
        best_length = current_length
        temp = initial_temp
        start_time = time.time()

        iteration = 1
        while temp > 1e-3:
            # generate neighbor: swap two
            i, j = random.sample(range(n), 2)
            neighbor = current_solution[:]
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbor_length = tour_length(neighbor)
            delta = neighbor_length - current_length
            # acceptance criterion
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_solution = neighbor
                current_length = neighbor_length
                if current_length < best_length:
                    best_solution = current_solution[:]
                    best_length = current_length
            # update temperature
            if cauchy:
                temp = initial_temp / (1 + cooling_rate * iteration)
            else:
                temp *= cooling_rate
            iteration += 1

        elapsed = time.time() - start_time
        # close cycle
        best_solution.append(best_solution[0])
        return best_solution, best_length, elapsed

    def solve_tsp_aco(self, num_ants=20, num_iters=100, alpha=1.0, beta=5.0, rho=0.5, wandering=True):
        """
        Ant Colony Optimization для TSP.
        если wandering=True — каждый муравей стартует из случайной вершины.
        """
        n = len(self.vertices)
        if n < 2:
            return None, float('inf'), 0

        # 1) Матрица расстояний
        dist = [[math.inf]*n for _ in range(n)]
        for i, j, d in self.edges:
            dist[i][j] = d

        # 2) Инициализация феромона
        tau0 = 1.0 / (n * sum(min(row) for row in dist if min(row) < math.inf))
        tau = [[tau0]*n for _ in range(n)]

        # 3) Эвристика η = 1/d
        eta = [[0 if dist[i][j]==math.inf else 1.0/dist[i][j] for j in range(n)] for i in range(n)]

        best_path, best_len = None, float('inf')
        start_time = time.time()

        for _ in range(num_iters):
            all_paths = []
            for _ in range(num_ants):
                # стартовая вершина
                current = random.randrange(n) if wandering else 0
                visited = {current}
                path = [current]

                while len(path) < n:
                    i = path[-1]
                    # вероятности перехода
                    probs = []
                    for j in range(n):
                        if j not in visited and dist[i][j] < math.inf:
                            probs.append(((i,j), (tau[i][j]**alpha)*(eta[i][j]**beta)))
                    if not probs:
                        break
                    total = sum(p for (_,p) in probs)
                    r = random.random() * total
                    s = 0
                    for (edge,p) in probs:
                        s += p
                        if s >= r:
                            nxt = edge[1]
                            break
                    visited.add(nxt)
                    path.append(nxt)

                # замкнём цикл
                if len(path)==n and dist[path[-1]][path[0]]<math.inf:
                    path.append(path[0])
                    length = sum(dist[path[k]][path[k+1]] for k in range(n))
                    all_paths.append((path, length))

                    if length < best_len:
                        best_len = length
                        best_path = path[:]

            # испарение
            for i in range(n):
                for j in range(n):
                    tau[i][j] *= (1 - rho)
            # обновление феромона
            for path, length in all_paths:
                delta = 1.0 / length
                for k in range(len(path)-1):
                    i,j = path[k], path[k+1]
                    tau[i][j] += delta

        elapsed = time.time() - start_time
        return best_path, best_len, elapsed


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
GraphCanvas.load_graph_from_json = load_graph_from_json

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
GraphCanvas.generate_random_graph = generate_random_graph

def solve_tsp_with_timing(self):
    vertices = self.graph_canvas.vertices
    edges = self.graph_canvas.edges
    n = len(vertices)
    self.txt_result.delete("1.0", tk.END)
    if n < 2:
        self.txt_result.insert(tk.END, "Недостаточно вершин для решения.\n")
        return

    dist_matrix = [[math.inf]*n for _ in range(n)]
    for (i, j, dist) in edges:
        dist_matrix[i][j] = dist

    start_time = time.time()
    start = 0
    visited = [False]*n
    visited[start] = True
    path = [start]
    total_dist = 0.0
    current = start
    for _ in range(n - 1):
        next_vertex = None
        min_dist = math.inf
        for j in range(n):
            if not visited[j] and dist_matrix[current][j] < min_dist:
                min_dist = dist_matrix[current][j]
                next_vertex = j
        if next_vertex is None:
            break
        visited[next_vertex] = True
        path.append(next_vertex)
        total_dist += min_dist
        current = next_vertex

    can_close_cycle = (dist_matrix[current][start] < math.inf)
    if can_close_cycle:
        total_dist += dist_matrix[current][start]
        path.append(start)
    all_visited = all(visited)
    end_time = time.time()

    self.txt_result.insert(tk.END, f"Порядок обхода: {path}\\n")
    self.txt_result.insert(tk.END, f"Длина пути: {total_dist:.2f}\\n")
    self.txt_result.insert(tk.END, f"Время выполнения: {end_time - start_time:.6f} сек\\n")
    if not all_visited or not can_close_cycle:
        self.txt_result.insert(tk.END, "Предупреждение: Гамильтонов цикл не найден. Путь неполный.\\n")

class TSPApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TSP Simulated Annealing")
        self.drawing_config = GraphConfig()
        
        self.drawing_config = GraphConfig()
        frame_main = tk.Frame(self)
        frame_main.pack(fill=tk.BOTH, expand=True)

        # Верхняя часть: граф и панель управления
        frame_top = tk.Frame(frame_main)
        frame_top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.graph_canvas = GraphCanvas(frame_top, config=self.drawing_config, bg="white")
        self.graph_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        frame_right = tk.Frame(frame_top)
        frame_right.pack(side=tk.RIGHT, fill=tk.Y)

        # Нижняя часть: таблица результатов
        frame_bottom = tk.Frame(frame_main)
        frame_bottom.pack(side=tk.BOTTOM, fill=tk.X)

        self.txt_result_label = tk.Text(frame_bottom, height=1, width=40)
        self.txt_result_label.pack(pady=2, padx=5, anchor="w")
        self.txt_result_label.insert(tk.END, "Результаты (Таблица):")
        self.txt_result_label.config(state=tk.DISABLED)  # Делаем текст нередактируемым

        self.txt_result = tk.Text(frame_bottom, height=8)
        self.txt_result.pack(fill=tk.X, padx=5, pady=5)
        
        # btn_sa = tk.Button(frame_right, text="SA TSP", command=self.on_solve_sa)
        # btn_sa.pack(pady=5, padx=5)
        # btn_sa_cauchy = tk.Button(frame_right, text="SA TSP (Cauchy)", command=lambda: self.on_solve_sa(cauchy=True))
        # btn_sa_cauchy.pack(pady=5, padx=5)

        


        self.var_directed = tk.BooleanVar(value=True)
        chk_directed = tk.Checkbutton(frame_right, text="Ориентированный граф",
            variable=self.var_directed, command=self.update_canvas_options)
        chk_directed.pack(pady=5, padx=5)

        self.var_weighted = tk.BooleanVar(value=True)
        chk_weighted = tk.Checkbutton(frame_right, text="Взвешанный граф",
            variable=self.var_weighted, command=self.update_canvas_options)
        chk_weighted.pack(pady=5, padx=5)

        lbl_weight = tk.Label(frame_right, text="Вес для нового ребра:")
        lbl_weight.pack(pady=2, padx=5)
        self.entry_weight = tk.Entry(frame_right)
        self.entry_weight.pack(pady=2, padx=5)
        self.entry_weight.insert(0, "3.0")

        self.btn_toggle_edge = tk.Button(frame_right, text="Режим добавления рёбер", command=self.toggle_edge_mode)
        self.btn_toggle_edge.pack(pady=5, padx=5)
        
        self.selected_method = tk.StringVar(value="NN")
        self.sa_cooling    = tk.StringVar(value="exponential")  
        
                # сразу после существующих Radiobutton для SA:
        tk.Label(frame_right, text="Метод решения:").pack(pady=2, padx=5)
        tk.Radiobutton(frame_right, text="NN", variable=self.selected_method, value="NN").pack(anchor="w", padx=5)
        tk.Radiobutton(frame_right, text="SA", variable=self.selected_method, value="SA").pack(anchor="w", padx=5)
        tk.Radiobutton(frame_right, text="ACO", variable=self.selected_method, value="ACO").pack(anchor="w", padx=5)

        # флажок блуждающей колонии (для ACO)
        self.aco_wandering = tk.BooleanVar(value=True)
        tk.Checkbutton(frame_right, text="Блуждающая колония", variable=self.aco_wandering).pack(anchor="w", padx=5)

        tk.Label(frame_right, text="Охлаждение (SA):").pack(pady=2, padx=5)
        tk.Radiobutton(frame_right, text="Эксп.", variable=self.sa_cooling, value="exponential").pack(anchor="w", padx=5)
        tk.Radiobutton(frame_right, text="Коши",   variable=self.sa_cooling, value="cauchy").pack(anchor="w", padx=5)

        # вставьте прямо под чекбоксом aco_wandering
        btn_aco = tk.Button(frame_right, text="Решить ACO", 
            command=lambda: self.on_solve_aco(wandering=False))
        btn_aco.pack(fill="x", padx=5, pady=2)

        btn_aco_w = tk.Button(frame_right, text="Решить ACO (блужд.)", 
            command=lambda: self.on_solve_aco(wandering=True))
        btn_aco_w.pack(fill="x", padx=5, pady=2)
                

        self.btn_clear = tk.Button(frame_right, text="Очистить граф", command=self.clear_graph)
        self.btn_clear.pack(pady=5, padx=5)
        
        self.btn_solve5 = tk.Button(frame_right, text="Решить 5 раз", command=self.solve_tsp_5_times)
        self.btn_solve5.pack(pady=5, padx=5)


        self.btn_solve = tk.Button(frame_right, text="Решить (ближ. сосед)", command=self.solve_tsp)
        self.btn_solve.pack(pady=5, padx=5)

        self.btn_example = tk.Button(frame_right, text="Загрузить контрольный пример", command=self.load_control_example)
        self.btn_example.pack(pady=5, padx=5)

        self.btn_show_matrix = tk.Button(frame_right, text="Показать матрицу смежности", command=self.show_adjacency_matrix)
        self.btn_show_matrix.pack(pady=5, padx=5)

        self.btn_load_json = tk.Button(frame_right, text="Загрузить граф из JSON", command=self.load_graph_json)
        self.btn_load_json.pack(pady=5, padx=5)

        self.btn_save_json = tk.Button(frame_right, text="Сохранить граф в JSON", command=self.save_graph_json)
        self.btn_save_json.pack(pady=5, padx=5)

        self.btn_random = tk.Button(frame_right, text="Случайный граф", command=self.generate_random_graph)
        self.btn_random.pack(pady=5, padx=5)

        # self.lbl_result = tk.Label(frame_right, text="Результат:")
        # self.lbl_result.pack(pady=5, padx=5)
        # self.txt_result = tk.Text(frame_right, height=10, width=30)
        # self.txt_result.pack(pady=5, padx=5)

        self.graph_canvas.var_directed = self.var_directed
        self.graph_canvas.var_weighted = self.var_weighted
        self.graph_canvas.entry_weight = self.entry_weight

    def update_canvas_options(self):
        self.graph_canvas.redraw()

    def toggle_edge_mode(self):
        mode = not self.graph_canvas.add_edge_mode
        self.graph_canvas.set_add_edge_mode(mode)
        self.btn_toggle_edge.config(text="Выключить режим рёбер" if mode else "Режим добавления рёбер")

    def clear_graph(self):
        self.graph_canvas.clear_graph()
        self.txt_result.delete("1.0", tk.END)

    def load_control_example(self):
        self.graph_canvas.clear_graph()
        coords = [(100,100),(200,80),(300,120),(250,200),(150,220),(100,180)]
        for i,(x,y) in enumerate(coords):
            self.graph_canvas.vertices.append({"x":x,"y":y,"name":str(i)})
        n = len(self.graph_canvas.vertices)
        for i in range(n):
            for j in range(i+1,n):
                dist = math.hypot(
                    self.graph_canvas.vertices[i]["x"] - self.graph_canvas.vertices[j]["x"],
                    self.graph_canvas.vertices[i]["y"] - self.graph_canvas.vertices[j]["y"]
                )
                self.graph_canvas.edges.append((i,j,dist))
                if not self.var_directed.get():
                    self.graph_canvas.edges.append((j,i,dist))
        self.graph_canvas.redraw()

    def on_solve_aco(self, wandering=False):
        sol, length, elapsed = self.graph_canvas.solve_tsp_aco(
            num_ants=20, num_iters=100,
            alpha=1.0, beta=5.0, rho=0.5,
            wandering=wandering
        )
        self.txt_result.delete("1.0", tk.END)
        if sol is None:
            self.txt_result.insert(tk.END, "Недостаточно вершин для ACO.\n")
            return
        self.txt_result.insert(tk.END, f"[ACO]{' блужд.' if wandering else ''} Путь: {sol}\n")
        self.txt_result.insert(tk.END, f"Длина: {length:.2f}\n")
        self.txt_result.insert(tk.END, f"Время: {elapsed:.6f} сек\n")

    def solve_tsp(self):
        vertices = self.graph_canvas.vertices
        edges = self.graph_canvas.edges
        n = len(vertices)
        self.txt_result.delete("1.0", tk.END)
        if n < 2:
            self.txt_result.insert(tk.END, "Недостаточно вершин для решения.\n")
            return
        dist_matrix = [[math.inf]*n for _ in range(n)]
        for (i, j, dist) in edges:
            dist_matrix[i][j] = dist
        start = 0
        visited = [False]*n
        visited[start] = True
        path = [start]
        total_dist = 0.0
        current = start
        for _ in range(n - 1):
            next_vertex = None
            min_dist = math.inf
            for j in range(n):
                if not visited[j] and dist_matrix[current][j] < min_dist:
                    min_dist = dist_matrix[current][j]
                    next_vertex = j
            if next_vertex is None:
                break
            visited[next_vertex] = True
            path.append(next_vertex)
            total_dist += min_dist
            current = next_vertex
        can_close_cycle = (dist_matrix[current][start] < math.inf)
        if can_close_cycle:
            total_dist += dist_matrix[current][start]
            path.append(start)
        all_visited = all(visited)
        self.txt_result.insert(tk.END, f"Порядок обхода: {path}\n")
        self.txt_result.insert(tk.END, f"Длина пути: {total_dist:.2f}\n")
        if not all_visited or not can_close_cycle:
            self.txt_result.insert(tk.END, "Предупреждение: Гамильтонов цикл не найден. Путь неполный.\n")
    
    def solve_tsp_with_timing(self):
        vertices = self.graph_canvas.vertices
        edges = self.graph_canvas.edges
        n = len(vertices)
        if n < 2:
            return "Недостаточно вершин для решения.\n"

        dist_matrix = [[math.inf]*n for _ in range(n)]
        for (i, j, dist) in edges:
            dist_matrix[i][j] = dist

        start_time = time.time()
        start = 0
        visited = [False]*n
        visited[start] = True
        path = [start]
        total_dist = 0.0
        current = start
        for _ in range(n - 1):
            next_vertex = None
            min_dist = math.inf
            for j in range(n):
                if not visited[j] and dist_matrix[current][j] < min_dist:
                    min_dist = dist_matrix[current][j]
                    next_vertex = j
            if next_vertex is None:
                break
            visited[next_vertex] = True
            path.append(next_vertex)
            total_dist += min_dist
            current = next_vertex

        can_close_cycle = (dist_matrix[current][start] < math.inf)
        if can_close_cycle:
            total_dist += dist_matrix[current][start]
            path.append(start)
        all_visited = all(visited)
        end_time = time.time()

        result = f"Порядок обхода: {path}\\n"
        result += f"Длина пути: {total_dist:.2f}\\n"
        result += f"Время выполнения: {end_time - start_time:.6f} сек\\n"
        if not all_visited or not can_close_cycle:
            result += "Предупреждение: Гамильтонов цикл не найден. Путь неполный.\\n"
        return result
    
    def run_nn(self, dist_matrix):
        n = len(dist_matrix)
        visited = [False]*n
        visited[0] = True
        path = [0]
        total = 0.0
        current = 0
        for _ in range(n-1):
            nxt, md = min(
                ((j, dist_matrix[current][j]) for j in range(n) if not visited[j]),
                key=lambda x: x[1], default=(None, None)
            )
            if nxt is None: break
            visited[nxt] = True
            path.append(nxt)
            total += md
            current = nxt
        if dist_matrix[current][0] < math.inf:
            total += dist_matrix[current][0]
            path.append(0)
        return path, total

    def run_nn_dist(self):
        # строим матрицу расстояний из графа
        verts = self.graph_canvas.vertices
        edges = self.graph_canvas.edges
        n = len(verts)
        dist = [[math.inf]*n for _ in range(n)]
        for i,j,d in edges:
            dist[i][j] = d

        # nearest neighbor
        visited = [False]*n
        visited[0] = True
        path = [0]
        total = 0.0
        current = 0
        for _ in range(n-1):
            nxt, md = min(
                ((j, dist[current][j]) for j in range(n) if not visited[j]),
                key=lambda x: x[1], default=(None, None)
            )
            if nxt is None:
                break
            visited[nxt] = True
            path.append(nxt)
            total += md
            current = nxt
        # закрываем цикл
        if dist[current][0] < math.inf:
            total += dist[current][0]
            path.append(0)
        return path, total



    def solve_tsp_5_times(self):
        self.txt_result.delete("1.0", tk.END)
        method = self.selected_method.get()
        use_cauchy = self.sa_cooling.get()
        header = f"{'#':<3}{'Путь':<40}{'Длина':<10}{'Время':<10}\n"
        self.txt_result.insert(tk.END, header + "-"*70 + "\n")

        for idx in range(5):
            self.generate_random_graph()
            n = len(self.graph_canvas.vertices)
            if n < 2:
                self.txt_result.insert(tk.END, f"{idx+1:<3}Недостаточно вершин\n")
                continue

            if method == "NN":
                path, length = self.run_nn_dist()
                elapsed = 0.0
            elif method == "SA":
                path, length, elapsed = self.graph_canvas.solve_tsp_simulated_annealing(
                    initial_temp=1000,
                    cooling_rate=0.995,
                    cauchy=(self.sa_cooling.get()=="cauchy")
                )
            else:  # ACO
                path, length, elapsed = self.graph_canvas.solve_tsp_aco(
                    num_ants=20,
                    num_iters=100,
                    alpha=1.0,
                    beta=5.0,
                    rho=0.5,
                    wandering=self.aco_wandering.get()
                )


            row = f"{idx+1:<3}{str(path):<40}{length:<10.2f}{elapsed:<10.4f}\n"
            self.txt_result.insert(tk.END, row)
            self.update()
            time.sleep(0.2)

    def show_adjacency_matrix(self):
        vertices = self.graph_canvas.vertices
        edges = self.graph_canvas.edges
        n = len(vertices)
        dist_matrix = [[math.inf]*n for _ in range(n)]
        for (i, j, dist) in edges:
            dist_matrix[i][j] = dist
        matrix_window = tk.Toplevel(self)
        matrix_window.title("Матрица смежности")
        info_label = tk.Label(matrix_window, text=f"Матрица смежности ({n}x{n}):")
        info_label.grid(row=0, column=0, columnspan=n+1, padx=5, pady=5)
        for col in range(n):
            label = tk.Label(matrix_window, text=str(col), bg="#ddd", width=5, relief="ridge")
            label.grid(row=1, column=col+1, sticky="nsew")
        for i in range(n):
            row_label = tk.Label(matrix_window, text=str(i), bg="#ddd", width=5, relief="ridge")
            row_label.grid(row=i+2, column=0, sticky="nsew")
            for j in range(n):
                val = dist_matrix[i][j]
                text_val = "∞" if val == math.inf else f"{val:.1f}"
                cell = tk.Label(matrix_window, text=text_val, width=5, relief="ridge")
                cell.grid(row=i+2, column=j+1, sticky="nsew")


    def show_adjacency_matrix(self):
        vertices = self.graph_canvas.vertices
        edges = self.graph_canvas.edges
        n = len(vertices)
        dist_matrix = [[math.inf]*n for _ in range(n)]
        for (i, j, dist) in edges:
            dist_matrix[i][j] = dist
        matrix_window = tk.Toplevel(self)
        matrix_window.title("Матрица смежности")
        info_label = tk.Label(matrix_window, text=f"Матрица смежности ({n}x{n}):")
        info_label.grid(row=0, column=0, columnspan=n+1, padx=5, pady=5)
        for col in range(n):
            label = tk.Label(matrix_window, text=str(col), bg="#ddd", width=5, relief="ridge")
            label.grid(row=1, column=col+1, sticky="nsew")
        for i in range(n):
            row_label = tk.Label(matrix_window, text=str(i), bg="#ddd", width=5, relief="ridge")
            row_label.grid(row=i+2, column=0, sticky="nsew")
            for j in range(n):
                val = dist_matrix[i][j]
                text_val = "∞" if val == math.inf else f"{val:.1f}"
                cell = tk.Label(matrix_window, text=text_val, width=5, relief="ridge")
                cell.grid(row=i+2, column=j+1, sticky="nsew")

    def load_graph_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON файлы", "*.json")])
        if path:
            self.graph_canvas.load_graph_from_json(path)

    def save_graph_json(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON файлы", "*.json")])
        if path:
            try:
                data = {
                    "vertices": self.graph_canvas.vertices,
                    "edges": self.graph_canvas.edges
                }
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                messagebox.showinfo("Успех", "Граф успешно сохранён.")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при сохранении:\n{e}")

    def generate_random_graph(self):
        self.graph_canvas.generate_random_graph(num_vertices=6, edge_prob=0.5, max_weight=15)
   
    def on_solve_sa(self, cauchy=False):
        sol, length, elapsed = self.graph_canvas.solve_tsp_simulated_annealing(
            initial_temp=1000,
            cooling_rate=0.995,
            cauchy=cauchy
        )
        self.txt_result.delete("1.0", tk.END)
        if sol is None:
            self.txt_result.insert(tk.END, "Недостаточно вершин для решения.\n")
            return
        self.txt_result.insert(tk.END, f"Порядок обхода: {sol}\n")
        self.txt_result.insert(tk.END, f"Длина пути: {length:.2f}\n")
        self.txt_result.insert(tk.END, f"Время выполнения: {elapsed:.6f} сек\n")


def run_5_times():
    app = TSPApp()
    app.withdraw()  # Скрыть окно
    for i in range(5):
        print(f"Прогон #{i + 1}")
        app.generate_random_graph()
        app.solve_tsp_with_timing()
        print(app.txt_result.get("1.0", tk.END))
        time.sleep(1)


def main():
    app = TSPApp()
    app.geometry("1000x600")
    app.mainloop()

if __name__ == "__main__":
    main()