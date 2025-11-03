import tkinter as tk
from tkinter import ttk

class ResultsWindow:
    def __init__(self, parent, results, path):
        self.window = tk.Toplevel(parent)
        self.window.title("Результаты тестирования")
        
        # Создаем таблицу
        frame = ttk.Frame(self.window, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Заголовки столбцов
        if path:
            headers = ["Тест", "Алгоритм", "Путь", "Длина", "Время"]
        else:
            headers = ["Тест", "Алгоритм", "Длина", "Время"]    
        
        # Заголовки
        for col, header in enumerate(headers):
            ttk.Label(frame, text=header, font=("Arial", 10, "bold")).grid(
                row=0, column=col, padx=5, pady=5, sticky="w"
            )
        
        # Настройка растягивания столбцов
        for col in range(len(headers)):
            frame.grid_columnconfigure(col, weight=1, uniform="equal")
        
        # Заполняем таблицу
        row = 1
        for test_name, algorithms in results.items():
            # Название теста
            ttk.Label(frame, text=test_name).grid(
                row=row, column=0, padx=5, pady=2, sticky="w"
            )
            
            # Результаты алгоритмов
            for alg_name, data in algorithms.items():
                ttk.Label(frame, text=alg_name).grid(
                    row=row, column=1, padx=5, pady=2, sticky="w"
                )
                if path:
                    ttk.Label(frame, text=data["path"]).grid(
                        row=row, column=2, padx=5, pady=2, sticky="w"
                    )
                ttk.Label(frame, text=f"{data['cost']:.2f}").grid(
                    row=row, column=3, padx=5, pady=2, sticky="w"
                )
                ttk.Label(frame, text=f"{data['time']:.6f}").grid(
                    row=row, column=4, padx=5, pady=2, sticky="w"
                )
                row += 1
            
            # Разделитель между тестами
            ttk.Separator(frame, orient="horizontal").grid(
                row=row, column=0, columnspan=5, sticky="ew", pady=5
            )
            row += 1
        
        # Кнопка закрытия
        ttk.Button(frame, text="Закрыть", command=self.window.destroy).grid(
            row=row, column=0, columnspan=5, pady=10
        )  
