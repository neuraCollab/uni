import tkinter as tk
from tkinter import ttk, messagebox
import random
import time  # Импортируем модуль time


def objective_function(individual):
    """Целевая функция."""
    return 4 * (individual[0] - 5) ** 2 + (individual[1] - 6) ** 2


def initialize_population(population_size, num_genes, min_gene, max_gene):
    """Создание начальной популяции."""
    return [[random.uniform(min_gene, max_gene) for _ in range(num_genes)] for _ in range(population_size)]


def select_parents(population, objective_function):
    """Рулеточный выбор родителей."""
    fitness_values = [1 / (1 + objective_function(ind)) for ind in population]
    total_fitness = sum(fitness_values)
    probabilities = [f / total_fitness for f in fitness_values]

    parents = []
    for _ in range(len(population)):
        r = random.random()
        cumulative = 0
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if r <= cumulative:
                parents.append(population[i])
                break
    return parents


def create_offspring(parent1, parent2):
    """Создание потомков через среднее арифметическое."""
    child1 = [0.5 * (gene1 + gene2) for gene1, gene2 in zip(parent1, parent2)]
    child2 = child1.copy()
    return child1, child2


def mutate(individual, mutation_rate, min_gene, max_gene):
    """Мутация индивида."""
    return [
        random.uniform(min_gene, max_gene) if random.random() < mutation_rate else gene
        for gene in individual
    ]


def evolve(population, num_generations, mutation_rate, min_gene, max_gene, modified, objective_function):
    """Основной процесс эволюции."""
    for _ in range(num_generations):
        if modified == "Enable":
            parents = select_parents(population, objective_function)
        else:
            parents = random.sample(population, len(population))

        new_population = []
        for _ in range(len(population) // 2):
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = create_offspring(parent1, parent2)
            new_population.extend([mutate(child1, mutation_rate, min_gene, max_gene),
                                   mutate(child2, mutation_rate, min_gene, max_gene)])
        population = new_population

    best = min(population, key=objective_function)
    fitness_values = [objective_function(ind) for ind in population]
    return population, best, objective_function(best), fitness_values


class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Evolutionary Algorithm")

        self.population_size = tk.IntVar(value=100)
        self.mutation_rate = tk.DoubleVar(value=0.1)
        self.modified = tk.StringVar(value="Enable")
        self.num_generations = tk.IntVar(value=50)
        self.min_gene = tk.DoubleVar(value=-10)
        self.max_gene = tk.DoubleVar(value=10)

        # Время выполнения
        self.execution_time = tk.StringVar(value="")

        self.setup_ui()

    def setup_ui(self):
        frame = ttk.LabelFrame(self.master, text="Parameters")
        frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Label(frame, text="Function: 4 * (x1 - 5)^2 + (x2 - 6)^2").grid(row=0, column=0, sticky="w")

        ttk.Label(frame, text="Population Size:").grid(row=1, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.population_size).grid(row=1, column=1, sticky="w")

        ttk.Label(frame, text="Mutation Rate:").grid(row=2, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.mutation_rate).grid(row=2, column=1, sticky="w")

        ttk.Label(frame, text="Min Gene Value:").grid(row=3, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.min_gene).grid(row=3, column=1, sticky="w")

        ttk.Label(frame, text="Max Gene Value:").grid(row=4, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.max_gene).grid(row=4, column=1, sticky="w")

        ttk.Label(frame, text="Generations:").grid(row=5, column=0, sticky="w")
        self.generations_entry = ttk.Entry(frame, textvariable=self.num_generations)
        self.generations_entry.grid(row=5, column=1, sticky="w")

        ttk.Label(frame, text="Modified Selection:").grid(row=6, column=0, sticky="w")
        ttk.Combobox(frame, textvariable=self.modified, values=["Enable", "Disable"]).grid(row=6, column=1, sticky="w")

        start_button = ttk.Button(frame, text="Start", command=self.run_algorithm)
        start_button.grid(row=7, column=0, columnspan=2, pady=5)

        results_frame = ttk.LabelFrame(self.master, text="Results")
        results_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        columns = ("Index", "Genes", "Fitness")
        self.tree = ttk.Treeview(results_frame, columns=columns, show="headings")
        for col in columns:
            self.tree.heading(col, text=col)
        self.tree.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)

        best_frame = ttk.LabelFrame(self.master, text="Best Individual")
        best_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        ttk.Label(best_frame, text="Best Genes:").grid(row=0, column=0, sticky="w")
        self.best_label = ttk.Label(best_frame, text="")
        self.best_label.grid(row=0, column=1, sticky="w")

        ttk.Label(best_frame, text="Best Fitness:").grid(row=1, column=0, sticky="w")
        self.fitness_label = ttk.Label(best_frame, text="")
        self.fitness_label.grid(row=1, column=1, sticky="w")

        ttk.Label(best_frame, text="Execution Time:").grid(row=2, column=0, sticky="w")
        self.time_label = ttk.Label(best_frame, textvariable=self.execution_time)
        self.time_label.grid(row=2, column=1, sticky="w")

    def update_results(self, population, fitness_values):
        for item in self.tree.get_children():
            self.tree.delete(item)

        for idx, (ind, fitness) in enumerate(zip(population, fitness_values), start=1):
            self.tree.insert("", "end", values=(idx, ind, fitness))

    def run_algorithm(self):
        try:
            # Увеличиваем количество итераций
            self.num_generations.set(self.num_generations.get() + 10)

            start_time = time.time()  # Запоминаем время начала
            population = initialize_population(self.population_size.get(), 2, self.min_gene.get(), self.max_gene.get())
            population, best, best_fitness, fitness_values = evolve(
                population,
                self.num_generations.get(),
                self.mutation_rate.get(),
                self.min_gene.get(),
                self.max_gene.get(),
                self.modified.get(),
                objective_function
            )
            end_time = time.time()  # Запоминаем время окончания

            execution_time = end_time - start_time  # Вычисляем время выполнения
            self.execution_time.set(f"{execution_time:.6f} seconds")  # Устанавливаем значение в метку

            self.update_results(population, fitness_values)
            self.best_label.config(text=f"{best}")
            self.fitness_label.config(text=f"{round(best_fitness, 6)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
