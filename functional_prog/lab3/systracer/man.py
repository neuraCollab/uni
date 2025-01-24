import psutil
import logging
import tkinter as tk
from tkinter import ttk, END, BOTH
import threading
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk as NavigationToolbar2TkAgg
from matplotlib.figure import Figure


logging.basicConfig(filename='system_audit.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
def monitor_processes():
    while True:
        yview = log_text.yview()
        log_text.delete(0, END)
        for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_times', 'memory_info']):
            try:
                pid = proc.info['pid']
                name = proc.info['name']
                username = proc.info['username']
                cpu_times = proc.info['cpu_times']
                memory_info = proc.info['memory_info']


                user_time = cpu_times.user
                system_time = cpu_times.system

                label = (f"PID: {pid}, Name: {name}, User: {username}, "
                        f"User Time: {user_time}s, System Time: {system_time}")

                log_message = (f"PID: {pid}, Name: {name}, User: {username}, "
                                f"User Time: {user_time}s, System Time: {system_time}s, "
                                f"Memory: {memory_info.rss / (1024 * 1024):.2f} MB, ")

                logging.info(log_message)

                log_text.insert(END, label)
                log_text.see(END)

                


            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        scrollbar.config(command=log_text.yview)
        time.sleep(5) 

sniffing_flag = False

def wrapper():
    global sniffing_flag
    if(sniffing_flag == False):
        sniffing_flag = True
        thread = threading.Thread(target = monitor_processes, daemon=True)
        thread.start()
        start_button.config(text="Остановить")
    else:
        sniffing_flag = False
        start_button.config(text="Запуск")

if __name__ == "__main__":
    root = tk.Tk()

    root = root
    root.geometry("1300x800")
    root.title("Мониторинг системы")



    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill=BOTH)

    frame1 = ttk.Frame(notebook)
    frame2 = ttk.Frame(notebook)

    left_frame = ttk.Frame(frame1)
    left_frame.grid(column=0, row=0, padx=100, pady=10, ipadx=6, ipady=6, sticky="nw")

    start_button = ttk.Button(left_frame, text="Запустить", command=wrapper)
    start_button.grid(column=0, row=0, padx=5, pady=5)

    right_frame = ttk.Frame(frame1)
    right_frame.grid(column=1, row=0, padx=100, pady=10, ipadx=6, ipady=6, sticky="ne")

    scrollbar = tk.Scrollbar(right_frame, orient=tk.VERTICAL)

    log_text = tk.Listbox(right_frame, yscrollcommand=scrollbar.set, width=100, height=70)
    log_text.grid(column=0, row=0, padx=5, pady=5)

    scrollbar.grid(column=1, row=0, sticky="ns")
    scrollbar.config(command=log_text.yview)

    notebook.add(frame1, text="Process")
    notebook.add(frame2, text="Statistics")


    root.mainloop()
