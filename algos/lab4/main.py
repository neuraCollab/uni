import tkinter as tk
from tkinter import ttk, messagebox

import random

class GeneticAlgorithm:
    def __init__(self, population_size, num_genes, num_generations, mutation_rate, min_gene, max_gene, encoding, modyfied):
        self.population_size = population_size
        self.num_genes = num_genes
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.min_gene = min_gene
        self.max_gene = max_gene
        self.encoding = encoding
        self.modyfied = modyfied
        self.population = self.create_population()


    def eval_func(self, individual):
        return 8 * individual[0] ** 2 + 4 * individual[0] * individual[1] + 5 * individual[1] ** 2

    def create_population(self):
        population = []
        for _ in range(self.population_size):
            if self.encoding == "real":
                individual = [random.uniform(self.min_gene, self.max_gene) for _ in range(self.num_genes)]
            else:
                individual = [random.randint(self.min_gene, self.max_gene) for _ in range(self.num_genes)]
            population.append(individual)
        return population

    def selection(self):
        parents = []
        for _ in range(self.population_size):
            tournament = random.sample(self.population, 3)
            tournament.sort(key=lambda x: self.eval_func(x))
            parents.append(tournament[0])
        return parents

    def crossover_modyfied(self, parents, mask):
        child1 = []
        child2 = []
        for gene1, gene2, bit in zip(parents[0], parents[1], mask):
            if self.encoding == "binary":
                binary_mask = int("".join([random.choice(["0", "1"]) for _ in range(max(len(bin(gene1)), len(bin(gene2))))]), 2)
                gene_1_with_mask = gene1 & binary_mask 
                gene_2_with_mask = gene2 & binary_mask 
                if bit == 0:
                    child1.append(gene_1_with_mask)
                    child2.append(gene_2_with_mask)
                else:
                    child1.append(gene_2_with_mask)
                    child2.append(gene_1_with_mask)
            elif self.encoding == "real":
                if bit == 0:
                    child1.append(gene1)
                    child2.append(gene2)
                else:
                    child1.append(gene2)
                    child2.append(gene1)

        return child1, child2
    def crossover_without_modyfied(self, parents, alpha = 0.5):
        child1 = []
        child2 = []
        for gene1, gene2 in zip(parents[0], parents[1]):
            child1.append(alpha * gene1 + (1 - alpha) * gene2)
            child2.append(alpha * gene2 + (1 - alpha) * gene1)
        return child1, child2

    def mutation(self, individual):
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                if self.encoding == "real":
                    individual[i] = random.uniform(self.min_gene, self.max_gene)
                elif self.encoding == "binary":
                    individual[i] = random.randint(self.min_gene, self.max_gene)
        return individual

    def generate_mask(self):
        return [random.randint(0, 1) for _ in range(self.num_genes)]

    def run(self):
        for _ in range(self.num_generations):
            parents = self.selection()
            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = random.sample(parents, 2)
                mask = self.generate_mask()
                if self.modyfied == "Включить":
                    child1, child2 = self.crossover_modyfied([parent1, parent2], mask)
                else:
                    child1, child2 = self.crossover_without_modyfied([parent1, parent2])

                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                new_population.append(child1)
                new_population.append(child2)
            self.population = new_population
        best_individual = min(self.population, key=lambda x: self.eval_func(x))
        individual_fitness = list(map(lambda x: self.eval_func(x), self.population))


        return self.population, best_individual, self.eval_func(best_individual), individual_fitness

class UI:
    def __init__(self, master):
        self.master = master
        self.master.title("Генетический алгоритм")

        # Переменные для хранения параметров генетического алгоритма
        self.population_size = tk.IntVar(value=100)
        self.mutation_rate = tk.DoubleVar(value=0.2)
        self.encoding = tk.StringVar(value='real')
        self.modyfied = tk.StringVar(value='Включить')
        self.iterations_entry = tk.IntVar(value=10)
        self.count_of_chomosomes = tk.IntVar(value=2)
        self.min_value_of_searching = tk.IntVar(value=-10)
        self.max_value_of_searching = tk.IntVar(value=10)
        self.iteration_count = tk.IntVar(value=10)
        self.iteration_count_done = 0

        # Создание и размещение элементов управления
        self.create_widgets()

    def create_widgets(self):
        # Фрейм для ввода параметров
        parameter_frame = ttk.LabelFrame(self.master, text="Параметры генетического алгоритма")
        parameter_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Label(parameter_frame, text="Функция: 8 * x ** 2 + 4 * x * y + 5 * y ** 2").grid(row=0, column=0, sticky="w")

        ttk.Label(parameter_frame, text="Размер популяции:").grid(row=1, column=0, sticky="w")
        population_entry = ttk.Entry(parameter_frame, textvariable=self.population_size)
        population_entry.grid(row=1, column=1, sticky="w")

        ttk.Label(parameter_frame, text="Вероятность мутации:").grid(row=2, column=0, sticky="w")
        mutation_rate_entry = ttk.Entry(parameter_frame, textvariable=self.mutation_rate)
        mutation_rate_entry.grid(row=2, column=1, sticky="w")

        ttk.Label(parameter_frame, text="Минимальное значение гена:").grid(row=3, column=0, sticky="w")
        mutation_rate_entry = ttk.Entry(parameter_frame, textvariable=self.min_value_of_searching)
        mutation_rate_entry.grid(row=3, column=1, sticky="w")

        ttk.Label(parameter_frame, text="Максимальное значение гена:").grid(row=4, column=0, sticky="w")
        mutation_rate_entry = ttk.Entry(parameter_frame, textvariable=self.max_value_of_searching)
        mutation_rate_entry.grid(row=4, column=1, sticky="w")

        ttk.Label(parameter_frame, text="Тип кодировки:").grid(row=5, column=0, sticky="w")
        encoding_combobox = ttk.Combobox(parameter_frame, textvariable=self.encoding, values=['real', 'binary'])
        encoding_combobox.grid(row=5, column=1, sticky="w")

        ttk.Label(parameter_frame, text="Модификация:").grid(row=6, column=0, sticky="w")
        modyfied_combobox = ttk.Combobox(parameter_frame, textvariable=self.modyfied, values=['Включить', 'Выключить'])
        modyfied_combobox.grid(row=6, column=1, sticky="w")


        ttk.Label(parameter_frame, text="Количество итераций:").grid(row=7, column=0, sticky="w")
        mutation_rate_entry = ttk.Entry(parameter_frame, textvariable=self.iteration_count)
        mutation_rate_entry.grid(row=7, column=1, sticky="w")


        # Фрейм для кнопок
        button_frame = ttk.Frame(self.master)
        button_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        start_button = ttk.Button(button_frame, text="Старт", command=self.start_algorithm)
        start_button.grid(row=0, column=0, padx=5)


        stat_frame = ttk.LabelFrame(self.master, text="Результаты")
        stat_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")


        # Создание таблицы для отображения популяции
        columns = ("Номер особи", "Геном особи", "Значение приспособленности")
        self.population_tree = ttk.Treeview(stat_frame, columns=columns, show="headings")
        for col in columns:
            self.population_tree.heading(col, text=col)
        self.population_tree.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Скроллбар для таблицы
        scrollbar = ttk.Scrollbar(stat_frame, orient="vertical", command=self.population_tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.population_tree.configure(yscrollcommand=scrollbar.set)

        # Фрейм для вывода лучшего результата
        best_result_frame = ttk.LabelFrame(self.master, text="Лучший результат")
        best_result_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Label(best_result_frame, text="Координаты точки:").grid(row=0, column=0, sticky="w")
        self.best_result_label = ttk.Label(best_result_frame, text="")
        self.best_result_label.grid(row=0, column=1, sticky="w")

        ttk.Label(best_result_frame, text="Значение функции:").grid(row=1, column=0, sticky="w")
        self.best_fitness_label = ttk.Label(best_result_frame, text="")
        self.best_fitness_label.grid(row=1, column=1, sticky="w")

        iteration_frame = ttk.Frame(self.master)
        iteration_frame.grid(row=4, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Label(iteration_frame, text="Количество итераций:").grid(row=0, column=0, sticky="w")
        self.iteration_label = ttk.Label(iteration_frame, text=self.iteration_count_done)
        self.iteration_label.grid(row=0, column=1, sticky="w")

    def set_table(self, population, individual_fitness):
        for item in self.population_tree.get_children():
            self.population_tree.delete(item)


        for i, individual in enumerate(population, start=1):
            self.population_tree.insert("", "end", values=(i, [round(individual[0], 4), round(individual[0], 4)], individual_fitness[i-1]))

    def start_algorithm(self):
        try:
            population_size = self.population_size.get()
            mutation_rate = self.mutation_rate.get()
            min_value_of_searching = self.min_value_of_searching.get()
            max_value_of_searching = self.max_value_of_searching.get()
            num_generations = self.iteration_count.get()
            encoding = self.encoding.get()
            modyfied = self.modyfied.get()
        except ValueError:
            messagebox.showerror("Ошибка", "В некоторые поля не были преданы аргументы")
            return

        self.iteration_count_done += self.iteration_count.get()

        ga = GeneticAlgorithm(population_size=100, num_genes=2, num_generations=num_generations, mutation_rate=mutation_rate, min_gene=min_value_of_searching, max_gene=max_value_of_searching, encoding=encoding, modyfied=modyfied)
        population, best_individual, best_fitness, individual_fitness = ga.run()
        self.set_table(population, individual_fitness)
        self.update_count_iterations(self.iteration_count_done)
        self.update_best_result_label(best_individual, best_fitness)

    def update_best_result_label(self, best_individual, best_fitness):
        print(f"({round(best_individual[0], 10)}, {round(best_individual[1], 10)})")
        print(round(best_fitness, 15))
        self.best_result_label.config(text=f"Координаты точки: {best_individual[0], best_individual[1]}")
        self.best_fitness_label.config(text=f"Значение функции: {round(best_fitness, 6)}")

    def update_count_iterations(self, iteration_count):
        self.iteration_label.config(text=f"{iteration_count}")


if __name__ == "__main__":
    root = tk.Tk()
    app = UI(root)
    root.mainloop()
