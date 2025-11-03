import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import os
from evaluation_gui import run_evaluation, generate_plots
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class ImputationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Оценка методов импутации")
        self.geometry("1400x900")

        self.plot_frame = None

        self.datasets = []  # Список (имя, DataFrame)
        self.current_dataset_index = tk.IntVar(value=0)
        self.methods = [
            "mean", "median", "mode", "spline",
            "linear_regression", "stochastic_regression"
        ]

        self._create_widgets()

    def _create_widgets(self):
        control_frame = tk.Frame(self)
        control_frame.pack(pady=10)

        # Кнопка загрузки одного CSV
        load_button = tk.Button(control_frame, text="Загрузить CSV", command=self.load_csv)
        load_button.grid(row=0, column=0, padx=5)

        # Кнопка загрузки из папки
        folder_button = tk.Button(control_frame, text="Оценить папку ./data", command=self.load_from_folder)
        folder_button.grid(row=0, column=1, padx=5)

        # Выбор датасета
        self.dataset_selector = ttk.Combobox(
            control_frame, textvariable=self.current_dataset_index, state="readonly"
        )
        self.dataset_selector.bind("<<ComboboxSelected>>", lambda e: self.update_output())
        self.dataset_selector.grid(row=0, column=2, padx=5)

        # Чекбоксы методов
        checkbox_frame = tk.Frame(control_frame)
        checkbox_frame.grid(row=1, column=0, columnspan=3, pady=5)

        self.method_vars = {}
        for i, method in enumerate(self.methods):
            var = tk.BooleanVar(value=True)
            chk = tk.Checkbutton(checkbox_frame, text=method, variable=var)
            chk.grid(row=0, column=i, padx=3)
            self.method_vars[method] = var

        # Кнопка запуска оценки
        run_button = tk.Button(control_frame, text="Запустить оценку", command=self.run_evaluation)
        run_button.grid(row=2, column=0, columnspan=3, pady=5)

        # Вывод
        self.output_frame = tk.Frame(self)
        self.output_frame.pack(fill=tk.BOTH, expand=True)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                df = pd.read_csv(file_path)
                name = os.path.basename(file_path)
                self.datasets.append((name, df))
                self.update_dataset_selector()
                messagebox.showinfo("Успех", f"Загружен: {name}")
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

    def load_from_folder(self):
        folder_path = "./data"
        if not os.path.exists(folder_path):
            messagebox.showerror("Ошибка", "Папка './data' не найдена.")
            return

        self.datasets.clear()
        loaded = 0
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(folder_path, filename)
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                    self.datasets.append((filename, df))
                    loaded += 1
                except Exception as e:
                    print(f"Ошибка при загрузке {filename}: {e}")

        if loaded == 0:
            messagebox.showwarning("Внимание", "В папке ./data нет корректных CSV.")
            return

        self.update_dataset_selector()
        self.evaluate_all_datasets()  # ⬅ Автоматический запуск анализа
        
    def evaluate_all_datasets(self):
        selected_methods = [m for m, var in self.method_vars.items() if var.get()]
        if not selected_methods or not self.datasets:
            return

        all_results = []
        for idx, (name, dataset) in enumerate(self.datasets):
            self.current_dataset_index.set(idx)
            summary, final_results, _ = run_evaluation(dataset, selected_methods)
            if not summary.empty:
                summary = summary.copy()
                summary["Dataset"] = name
                all_results.append(summary)

        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            self.plot_combined_results(combined_df)

    def update_dataset_selector(self):
        items = [f"{name}" for name, _ in self.datasets]
        self.dataset_selector["values"] = items
        self.dataset_selector.current(len(self.datasets) - 1)
        self.current_dataset_index.set(len(self.datasets) - 1)

    def run_evaluation(self):
        selected_methods = [m for m, var in self.method_vars.items() if var.get()]
        if not selected_methods:
            messagebox.showwarning("Предупреждение", "Выберите хотя бы один метод.")
            return

        if not self.datasets:
            messagebox.showwarning("Предупреждение", "Сначала загрузите хотя бы один датасет.")
            return

        self.update_output(selected_methods)

    def update_output(self, selected_methods=None):
        for widget in self.output_frame.winfo_children():
            widget.destroy()

        idx = self.current_dataset_index.get()
        if idx >= len(self.datasets):
            return

        name, dataset = self.datasets[idx]
        if selected_methods is None:
            selected_methods = [m for m, var in self.method_vars.items() if var.get()]

        summary, final_results, _ = run_evaluation(dataset, selected_methods)
        if summary.empty:
            messagebox.showinfo("Результат", f"Нет результатов для {name}")
            return

        # Таблица с прокруткой
        table_frame = tk.Frame(self.output_frame, width=600)
        table_frame.pack(side="left", fill="y", padx=5, pady=5)
        table_frame.pack_propagate(False)

        tree = ttk.Treeview(table_frame, columns=list(summary.columns), show="headings")
        for col in summary.columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor="center")
        for _, row in summary.iterrows():
            tree.insert("", "end", values=list(row))

        tree.pack(side="left", fill="both", expand=True)

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscroll=vsb.set, xscroll=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")

        # Графики
        if self.plot_frame is not None and self.plot_frame.winfo_exists():
            for widget in self.plot_frame.winfo_children():
                widget.destroy()
        else:
            self.plot_frame = tk.Frame(self.output_frame)
            self.plot_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        generate_plots(final_results, summary, self.plot_frame)
        
    
    def plot_combined_results(self, combined_df):
        # График сравнения методов по всем датасетам
        window = tk.Toplevel(self)
        window.title("Сравнение по всем датасетам")
        window.geometry("1000x600")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        sns.lineplot(
            data=combined_df,
            x="Missing%",
            y="MeanRelativeError%",
            hue="Method",
            style="Dataset",
            markers=True,
            dashes=False,
            ax=ax1
        )
        ax1.set_title("Ошибка по пропускам")

        sns.lineplot(
            data=combined_df,
            x="Missing%",
            y="TimeSeconds",
            hue="Method",
            style="Dataset",
            markers=True,
            dashes=True,
            ax=ax2
        )
        ax2.set_title("Время выполнения (сек)")

        fig.tight_layout()


        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)




if __name__ == "__main__":
    app = ImputationApp()
    app.mainloop()
