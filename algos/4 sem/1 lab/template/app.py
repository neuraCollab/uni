import tkinter as tk
from graph_canvas import GraphCanvas
from config import GraphConfig
from tsp_solver import solve_nearest_neighbor
import random
import json
import math

class TSPApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Задача о коммивояжёре (Метод ближайшего соседа)")
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

        self.lbl_result = tk.Label(frame_bottom, text="Результаты (Таблица):")
        self.lbl_result.pack(pady=2, padx=5, anchor="w")

        self.txt_result = tk.Text(frame_bottom, height=8)
        self.txt_result.pack(fill=tk.X, padx=5, pady=5)


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

    def solve_tsp_5_times(self):
        self.txt_result.delete("1.0", tk.END)
        header = f"{'Граф':<6}{'Без модификации':<55}{'С модификацией':<55}\n"
        self.txt_result.insert(tk.END, header)
        self.txt_result.insert(tk.END, "-" * 115 + "\n")

        for idx in range(5):
            self.generate_random_graph()

            def run_nn(start_index):
                vertices = self.graph_canvas.vertices
                edges = self.graph_canvas.edges
                n = len(vertices)
                dist_matrix = [[math.inf] * n for _ in range(n)]
                for (i, j, dist) in edges:
                    dist_matrix[i][j] = dist

                visited = [False] * n
                visited[start_index] = True
                path = [start_index]
                total_dist = 0.0
                current = start_index

                t0 = time.time()
                for _ in range(n - 1):
                    next_v = None
                    min_dist = math.inf
                    for j in range(n):
                        if not visited[j] and dist_matrix[current][j] < min_dist:
                            min_dist = dist_matrix[current][j]
                            next_v = j
                    if next_v is None:
                        break
                    visited[next_v] = True
                    path.append(next_v)
                    total_dist += min_dist
                    current = next_v

                can_close = dist_matrix[current][start_index] < math.inf
                if can_close:
                    total_dist += dist_matrix[current][start_index]
                    path.append(start_index)

                t1 = time.time()
                return path, total_dist, t1 - t0

            # Обычный (старт с 0)
            path1, dist1, time1 = run_nn(start_index=0)

            # Модифицированный (старт с рандома)
            start_random = random.randint(0, len(self.graph_canvas.vertices) - 1)
            path2, dist2, time2 = run_nn(start_index=start_random)

            row = f"{idx+1:<6}" \
                f"{str(path1)} {dist1:.2f}, {time1:.6f} сек,{'' if len(path1)<10 else ' ' * (55-len(str(path1))-30)}" \
                f"{str(path2)} {dist2:.2f}, {time2:.6f} сек\n"

            self.txt_result.insert(tk.END, row)
            self.update()
            time.sleep(0.3)

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
