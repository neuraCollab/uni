import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from graph_canvas import GraphCanvas
from graph_config import GraphConfig
from algorithms.nearest_neighbor import nearest_neighbor
from algorithms.simulated_annealing import simulated_annealing
from algorithms.sa2 import solve_tsp_simulated_annealing
from algorithms.ant_colony import ant_colony_optimization
from results_window import ResultsWindow
import time
import math

class TSPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Решение задачи коммивояжера")
        
        # Основные фреймы
        self.left_frame = ttk.Frame(root, padding="5")
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        self.right_frame = ttk.Frame(root, padding="5")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Канвас для графа
        self.canvas = GraphCanvas(self.right_frame, config=GraphConfig(), width=800, height=600, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Настройки графа
        self.setup_graph_controls()
        
        # Кнопки для запуска алгоритмов
        self.setup_algorithm_controls()
        
        # Кнопка для запуска тестов
        test_frame = ttk.LabelFrame(self.left_frame, text="Тестирование", padding="5")
        test_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(test_frame, text="Запустить все тесты", 
                  command=self.run_all_tests).pack(fill=tk.X, pady=5)
        
        # Результаты
        self.setup_results_area()
        
    def setup_graph_controls(self):
        graph_frame = ttk.LabelFrame(self.left_frame, text="Настройки графа", padding="5")
        graph_frame.pack(fill=tk.X, pady=5)
        
        # Режим добавления ребер
        self.var_add_edge = tk.BooleanVar()
        ttk.Checkbutton(graph_frame, text="Режим добавления ребер", 
                       variable=self.var_add_edge, 
                       command=self.toggle_add_edge_mode).pack(anchor=tk.W)
        
        # Ориентированный граф
        self.var_directed = tk.BooleanVar()
        ttk.Checkbutton(graph_frame, text="Ориентированный граф", 
                       variable=self.var_directed).pack(anchor=tk.W)
        self.canvas.var_directed = self.var_directed
        
        # Взвешенный граф
        self.var_weighted = tk.BooleanVar()
        ttk.Checkbutton(graph_frame, text="Взвешенный граф", 
                       variable=self.var_weighted).pack(anchor=tk.W)
        self.canvas.var_weighted = self.var_weighted
        
        # Вес ребра
        weight_frame = ttk.Frame(graph_frame)
        weight_frame.pack(fill=tk.X)
        ttk.Label(weight_frame, text="Вес ребра:").pack(side=tk.LEFT)
        self.entry_weight = ttk.Entry(weight_frame, width=10)
        self.entry_weight.pack(side=tk.LEFT, padx=5)
        self.entry_weight.insert(0, "3.0")
        self.canvas.entry_weight = self.entry_weight
        
        # Кнопки управления графом
        ttk.Button(graph_frame, text="Очистить граф", 
                  command=self.canvas.clear_graph).pack(fill=tk.X, pady=5)
        ttk.Button(graph_frame, text="Случайный граф", 
                  command=self.generate_random_graph).pack(fill=tk.X)
        ttk.Button(graph_frame, text="Загрузить контрольный пример", 
                  command=self.load_control_example).pack(fill=tk.X, pady=5)
        ttk.Button(graph_frame, text="Показать матрицу смежности", 
                  command=self.show_adjacency_matrix).pack(fill=tk.X)
        ttk.Button(graph_frame, text="Загрузить граф из JSON", 
                  command=self.load_graph_json).pack(fill=tk.X, pady=5)
        ttk.Button(graph_frame, text="Сохранить граф в JSON", 
                  command=self.save_graph_json).pack(fill=tk.X)
                   
    def setup_algorithm_controls(self):
        """Создает элементы управления для запуска алгоритмов"""
        alg_frame = ttk.LabelFrame(self.left_frame, text="Запуск алгоритма", padding="5")
        alg_frame.pack(fill=tk.X, pady=5)
        
        # Выбор алгоритма
        self.algorithm_var = tk.StringVar(value="nn")
        algorithms = [
            ("Ближайший сосед", "nn"),
            ("Имитация отжига", "sa"),
            ("Имитация отжига 2", "sa2"),
            ("Муравьиная колония", "aco")
        ]
        
        for text, value in algorithms:
            ttk.Radiobutton(alg_frame, text=text, value=value, 
                          variable=self.algorithm_var).pack(anchor=tk.W)
        
        # Настройки SA
        sa_frame = ttk.LabelFrame(alg_frame, text="Настройки SA", padding="5")
        sa_frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.sa_cooling = tk.StringVar(value="exponential")
        ttk.Radiobutton(sa_frame, text="Экспоненциальное охлаждение", 
                       value="exponential", variable=self.sa_cooling).pack(anchor=tk.W)
        ttk.Radiobutton(sa_frame, text="Охлаждение по Коши", 
                       value="cauchy", variable=self.sa_cooling).pack(anchor=tk.W)
        
        # Настройки ACO
        aco_frame = ttk.LabelFrame(alg_frame, text="Настройки ACO", padding="5")
        aco_frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.aco_wandering = tk.BooleanVar(value=True)
        ttk.Radiobutton(aco_frame, text="Блуждающая колония", 
                       value=True, variable=self.aco_wandering).pack(anchor=tk.W)
        ttk.Radiobutton(aco_frame, text="Фиксированное начало", 
                       value=False, variable=self.aco_wandering).pack(anchor=tk.W)
        
        # Параметры ACO
        aco_params_frame = ttk.Frame(aco_frame)
        aco_params_frame.pack(fill=tk.X, pady=5)
        
        # Количество муравьев
        ttk.Label(aco_params_frame, text="Муравьев:").grid(row=0, column=0, padx=5)
        self.aco_ants = ttk.Entry(aco_params_frame, width=5)
        self.aco_ants.insert(0, "20")
        self.aco_ants.grid(row=0, column=1, padx=5)
        
        # Количество итераций
        ttk.Label(aco_params_frame, text="Итераций:").grid(row=0, column=2, padx=5)
        self.aco_iters = ttk.Entry(aco_params_frame, width=5)
        self.aco_iters.insert(0, "100")
        self.aco_iters.grid(row=0, column=3, padx=5)
        
        # Кнопки запуска
        ttk.Button(alg_frame, text="Решить", 
                  command=self.solve_current_graph).pack(fill=tk.X, pady=5)
        ttk.Button(alg_frame, text="Решить 5 раз", 
                  command=self.solve_tsp_5_times).pack(fill=tk.X)
        
    def setup_results_area(self):
        results_frame = ttk.LabelFrame(self.left_frame, text="Результаты", padding="5")
        results_frame.pack(fill=tk.X, pady=5)
        
        self.result_text = tk.Text(results_frame, height=10, width=40)
        self.result_text.pack(fill=tk.BOTH)
        
    def load_test_cases(self):
        """Загружает тестовые случаи из файла или генерирует их"""
        test_cases = []
        
        # Тестовый случай 1: маленький граф
        test_cases.append({
            "name": "Маленький граф (5 вершин)",
            "vertices": [(100,100), (200,100), (150,200), (100,300), (200,300)],
            "edges": []
        })
        
        # Тестовый случай 2: средний граф
        test_cases.append({
            "name": "Средний граф (8 вершин)",
            "vertices": [(100,100), (200,100), (300,100), (300,200),
                       (300,300), (200,300), (100,300), (100,200)],
            "edges": []
        })
        
        # Тестовый случай 3: большой граф
        vertices = []
        for i in range(10):
            angle = 2 * math.pi * i / 10
            x = 200 + 150 * math.cos(angle)
            y = 200 + 150 * math.sin(angle)
            vertices.append((int(x), int(y)))
        test_cases.append({
            "name": "Большой граф (10 вершин)",
            "vertices": vertices,
            "edges": []
        })

        # Тестовый случай 4: граф-звезда
        center = (200, 200)
        radius = 150
        vertices = [center]  # центральная вершина
        for i in range(7):  # 7 вершин по кругу
            angle = 2 * math.pi * i / 7
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            vertices.append((int(x), int(y)))
        test_cases.append({
            "name": "Граф-звезда (8 вершин)",
            "vertices": vertices,
            "edges": []
        })

        # Тестовый случай 5: граф-решетка
        grid_size = 7  # 3x3 решетка
        spacing = 100
        start_x = 100
        start_y = 100
        vertices = []
        for i in range(grid_size):
            for j in range(grid_size):
                vertices.append((start_x + j * spacing, start_y + i * spacing))
        test_cases.append({
            "name": "Граф-решетка (49 вершин)",
            "vertices": vertices,
            "edges": []
        })
        
        # Генерируем рёбра для всех тестовых случаев
        for case in test_cases:
            vertices = case["vertices"]
            n = len(vertices)
            edges = []
            for i in range(n):
                for j in range(n):
                    if i != j:
                        x1, y1 = vertices[i]
                        x2, y2 = vertices[j]
                        dist = math.hypot(x2-x1, y2-y1)
                        edges.append((i, j, dist))
            case["edges"] = edges
            
        return test_cases
        
    def toggle_add_edge_mode(self):
        mode = self.var_add_edge.get()
        self.canvas.set_add_edge_mode(mode)
        
    def generate_random_graph(self):
        self.canvas.generate_random_graph(num_vertices=6, edge_prob=0.5, max_weight=15)
        
    def load_control_example(self):
        self.canvas.clear_graph()
        coords = [(100,100), (200,80), (300,120), (250,200), (150,220), (100,180)]
        for i, (x,y) in enumerate(coords):
            self.canvas.vertices.append({"x":x, "y":y, "name":str(i)})
        n = len(self.canvas.vertices)
        for i in range(n):
            for j in range(i+1, n):
                dist = math.hypot(
                    self.canvas.vertices[i]["x"] - self.canvas.vertices[j]["x"],
                    self.canvas.vertices[i]["y"] - self.canvas.vertices[j]["y"]
                )
                self.canvas.edges.append((i,j,dist))
                if not self.var_directed.get():
                    self.canvas.edges.append((j,i,dist))
        self.canvas.redraw()
        
    def show_adjacency_matrix(self):
        vertices = self.canvas.vertices
        edges = self.canvas.edges
        n = len(vertices)
        dist_matrix = [[float('inf')]*n for _ in range(n)]
        for (i, j, dist) in edges:
            dist_matrix[i][j] = dist
            
        matrix_window = tk.Toplevel(self.root)
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
                text_val = "∞" if val == float('inf') else f"{val:.1f}"
                cell = tk.Label(matrix_window, text=text_val, width=5, relief="ridge")
                cell.grid(row=i+2, column=j+1, sticky="nsew")
                
    def load_graph_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON файлы", "*.json")])
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.canvas.clear_graph()
                self.canvas.vertices = data.get("vertices", [])
                self.canvas.edges = [tuple(e) for e in data.get("edges", [])]
                self.canvas.redraw()
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить граф из файла:\n{e}")
                
    def save_graph_json(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", 
                                          filetypes=[("JSON файлы", "*.json")])
        if path:
            try:
                data = {
                    "vertices": self.canvas.vertices,
                    "edges": self.canvas.edges
                }
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                messagebox.showinfo("Успех", "Граф успешно сохранён.")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при сохранении:\n{e}")
                
    def solve_current_graph(self):
        """Решает текущий граф выбранным алгоритмом"""
        if not self.canvas.vertices:
            messagebox.showwarning("Предупреждение", "Создайте граф перед запуском алгоритма")
            return
            
        algorithm = self.algorithm_var.get()
        try:
            if algorithm == "nn":
                path, cost, time = nearest_neighbor(self.prepare_distances())
            elif algorithm == "sa":
                path, cost, time = self.canvas.solve_tsp_simulated_annealing(
                    cauchy=(self.sa_cooling.get()=="cauchy")
                )
            elif algorithm == "sa2":
                path, cost, time = solve_tsp_simulated_annealing(self.prepare_distances())
            else:  # aco
                try:
                    num_ants = int(self.aco_ants.get())
                    num_iters = int(self.aco_iters.get())
                except ValueError:
                    messagebox.showerror("Ошибка", "Количество муравьев и итераций должны быть целыми числами")
                    return
                    
                path, cost, time = self.canvas.solve_tsp_aco(
                    num_ants=num_ants,
                    num_iters=num_iters,
                    wandering=self.aco_wandering.get()
                )
                
            self.show_results(path, cost, time)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось решить задачу: {str(e)}")
            
    def solve_tsp_5_times(self):
        self.result_text.delete("1.0", tk.END)
        method = self.algorithm_var.get()
        header = f"{'#':<3}{'Путь':<40}{'Длина':<10}{'Время':<10}\n"
        self.result_text.insert(tk.END, header + "-"*70 + "\n")

        for idx in range(5):
            self.generate_random_graph()
            try:
                if method == "nn":
                    path, length, elapsed = nearest_neighbor(self.prepare_distances())
                elif method == "sa":
                    path, length, elapsed = self.canvas.solve_tsp_simulated_annealing(
                        cauchy=(self.sa_cooling.get()=="cauchy")
                    )
                elif method == "sa2":
                    path, length, elapsed = solve_tsp_simulated_annealing(self.prepare_distances())
                else:  # aco
                    try:
                        num_ants = int(self.aco_ants.get())
                        num_iters = int(self.aco_iters.get())
                    except ValueError:
                        messagebox.showerror("Ошибка", "Количество муравьев и итераций должны быть целыми числами")
                        return
                        
                    path, length, elapsed = self.canvas.solve_tsp_aco(
                        num_ants=num_ants,
                        num_iters=num_iters,
                        wandering=self.aco_wandering.get()
                    )

                row = f"{idx+1:<3}{str(path):<40}{length:<10.2f}{elapsed:<10.4f}\n"
                self.result_text.insert(tk.END, row)
                self.root.update()
                time.sleep(0.2)
                
            except Exception as e:
                self.result_text.insert(tk.END, f"{idx+1:<3}Ошибка: {str(e)}\n")
                
    def prepare_distances(self):
        n = len(self.canvas.vertices)
        distances = [[float('inf')]*n for _ in range(n)]
        for i, j, w in self.canvas.edges:
            distances[i][j] = w
        return distances
        
    def show_results(self, path, cost, time):
        # Очистка предыдущих результатов
        self.result_text.delete(1.0, tk.END)
        
        # Вывод нового маршрута
        path_str = " -> ".join(str(v) for v in path)
        self.result_text.insert(tk.END, f"Маршрут: {path_str}\n")
        self.result_text.insert(tk.END, f"Стоимость: {cost:.2f}\n")
        self.result_text.insert(tk.END, f"Время выполнения: {time:.6f} сек\n")
        
        # Подсветка маршрута на графе
        self.canvas.redraw()  # Сначала перерисовываем граф
        
        # Затем рисуем маршрут
        for i in range(len(path)-1):
            v1 = self.canvas.vertices[path[i]]
            v2 = self.canvas.vertices[path[i+1]]
            self.canvas.create_line(
                v1["x"], v1["y"],
                v2["x"], v2["y"],
                fill="red",
                width=2,
                dash=(4, 4)
            )

    def run_all_tests(self):
        """Запускает все алгоритмы на всех тестовых случаях"""
        test_cases = self.load_test_cases()
        if not test_cases:
            messagebox.showwarning("Предупреждение", "Нет доступных тестовых случаев")
            return
            
        # Очищаем текущий результат
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Запуск тестирования...\n\n")
        self.root.update()
        
        results = {}
        for test_case in test_cases:
            self.result_text.insert(tk.END, f"Тестирование: {test_case['name']}\n")
            self.root.update()
            
            # Подготовка графа
            self.canvas.clear_graph()
            for x, y in test_case["vertices"]:
                self.canvas.vertices.append({"x": x, "y": y, "name": str(len(self.canvas.vertices))})
            self.canvas.edges = test_case["edges"]
            self.canvas.redraw()
            
            # Получение матрицы расстояний
            distances = self.prepare_distances()
            
            # Запуск алгоритмов
            algorithms = {
                "Ближайший сосед": lambda: nearest_neighbor(distances, start_city=0),
                "Ближайший сосед с модификацией": lambda: nearest_neighbor(distances, start_city=0),
                "Имитация отжига": lambda: self.canvas.solve_tsp_simulated_annealing(
                    cauchy=(self.sa_cooling.get()=="cauchy")
                ),
                "Имитация отжига с модификацией": lambda: solve_tsp_simulated_annealing(distances),
                "Муравьиная колония": lambda: self.canvas.solve_tsp_aco(),
                "Муравьиная колония с модификацией": lambda: self.canvas.solve_tsp_aco(
                    wandering=self.aco_wandering.get()
                )
                
            }
            
            case_results = {}
            for name, func in algorithms.items():
                try:
                    path, cost, elapsed = func()
                    case_results[name] = {
                        "path": " -> ".join(str(v) for v in path),
                        "cost": cost,
                        "time": elapsed
                    }
                    self.result_text.insert(tk.END, 
                        f"{name}: длина={cost:.2f}, время={elapsed:.6f}с\n")
                except Exception as e:
                    case_results[name] = {
                        "path": "Ошибка",
                        "cost": float('inf'),
                        "time": 0
                    }
                    self.result_text.insert(tk.END, f"{name}: ошибка - {str(e)}\n")
                self.root.update()
                
            results[test_case["name"]] = case_results
            self.result_text.insert(tk.END, "\n")
            self.root.update()
            time.sleep(0.5)  # Пауза между тестами
            
        # Показываем окно с результатами
        ResultsWindow(self.root, results, False)

    def format_path(self, path):
        """Форматирует путь в виде строки"""
        if path is None:
            return "Путь не найден"
        return " -> ".join(str(v) for v in path)

if __name__ == "__main__":
    root = tk.Tk()
    app = TSPApp(root)
    root.geometry("1000x600")
    root.mainloop() 