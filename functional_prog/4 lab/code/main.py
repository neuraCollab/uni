import tkinter as tk
from tkinter import ttk, END
import json

rates_books = {}
books_info = {}
all_books = []

def load_library():
    with open("books.json", "r") as file:
        books = json.load(file)
    return books

def calc_like_rate(books):
    for book in books:
        books_info.update({book["title"]: book})
        if authors_enter.get() in book["author"]:
            if book["title"] not in rates_books.keys():
                rates_books.update({book["title"]: 10})
            else:
                rates_books[book["title"]] += 10
        if genr_enter.get() != "" and genr_enter.get() in book["genre"]:
            if book["title"] not in rates_books.keys():
                rates_books.update({book["title"]: 5})
            else:
                rates_books[book["title"]] += 5

        all_target_words = set(map(lambda x: x.lower(), main_words_enter.get().split(",")))
        all_words_in_descr = set(map(lambda x: x.lower(), book["description"].split()))
        intersec = all_target_words.intersection(all_words_in_descr)

        if intersec:
            if book["title"] not in rates_books.keys():
                rates_books.update({book["title"]: len(intersec)})
            else:
                rates_books[book["title"]] += len(intersec)

def update_list_books(all_books):
    books_listbox.delete(0, END)
    for book in all_books:
        books_listbox.insert(END, book)
    scrollbar.config(command=books_listbox.yview)

def find_books():
    global all_books
    all_books = []
    books = load_library()
    calc_like_rate(books)

    sorted_books = {}
    if choose_sort_by.get() == "По алфавиту":
        sorted_books = dict(sorted(rates_books.items(), key=lambda item: item[0]))
    elif choose_sort_by.get() == "По возрастанию рейтинга":
        sorted_books = dict(sorted(rates_books.items(), key=lambda item: item[1]))
    elif choose_sort_by.get() == "По убыванию рейтинга":
        sorted_books = dict(sorted(rates_books.items(), key=lambda item: item[1], reverse=True))
    elif choose_sort_by.get() == "По году публикации":
        sorted_books = dict(sorted(rates_books.items(), key=lambda item: books_info[item[0]]["first_publish_year"]))

    for title, _ in sorted_books.items():
        all_books.append(f"Название: {title}, Автор: {';'.join(books_info[title]['author'])}, Жанр: {books_info[title]['genre']}, Год публикации: {books_info[title]['first_publish_year']}")

    update_list_books(all_books)

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x800")
    root.title("Сервис рекомендаций книг")

    left_frame = ttk.Frame(root)
    left_frame.grid(column=0, row=0, padx=5, pady=10, ipadx=6, ipady=6, sticky="nw")

    authors = ttk.Label(left_frame, width=15, text="Авторы")
    authors.grid(column=0, row=0, padx=5, pady=5, sticky="w")
    authors_enter = ttk.Entry(left_frame)
    authors_enter.grid(column=0, row=1, padx=5, pady=5, sticky="w")

    genr = ttk.Label(left_frame, width=15, text="Жанры")
    genr.grid(column=0, row=2, padx=5, pady=5, sticky="w")
    genr_enter = ttk.Entry(left_frame)
    genr_enter.grid(column=0, row=3, padx=5, pady=5, sticky="w")

    main_words = ttk.Label(left_frame, width=15, text="Ключевые слова")
    main_words.grid(column=0, row=4, padx=5, pady=5, sticky="w")
    main_words_enter = ttk.Entry(left_frame)
    main_words_enter.grid(column=0, row=5, padx=5, pady=5, sticky="w")

    choose_sort_by_label = ttk.Label(left_frame, width=15, text="Сортировка")
    choose_sort_by_label.grid(column=0, row=6, padx=5, pady=5, sticky="w")
    choose_sort_by = ttk.Combobox(left_frame, values=["По алфавиту", "По возрастанию рейтинга", "По убыванию рейтинга", "По году публикации"])
    choose_sort_by.grid(column=0, row=7, padx=5, pady=5, sticky="w")
    choose_sort_by.current(0)

    find_books_button = ttk.Button(left_frame, text="Найти подходящие книги", command=find_books)
    find_books_button.grid(column=0, row=8, padx=5, pady=5, sticky="w")

    right_frame = ttk.Frame(root)
    right_frame.grid(column=2, row=0, padx=[250, 0], ipadx=6, ipady=6, sticky="ne")

    scrollbar = tk.Scrollbar(right_frame, orient=tk.VERTICAL)
    books_listbox = tk.Listbox(right_frame, yscrollcommand=scrollbar.set, width=100, height=30)
    books_listbox.grid(column=0, row=0, padx=5, pady=5)
    scrollbar.grid(column=1, row=0, sticky="ns")
    scrollbar.config(command=books_listbox.yview)

    root.mainloop()
