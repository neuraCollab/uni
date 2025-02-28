import psutil
import logging
import tkinter as tk
from tkinter import ttk, END, BOTH
import threading
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk as NavigationToolbar2TkAgg
from matplotlib.figure import Figure

# Настройка логирования
logging.basicConfig(filename='system_audit.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables to store process data
process_data = {'time': [], 'working': [], 'sleeping': [], 'zombie' : []}

def monitor_processes():
    global process_data
    while True:
        if stop_flag:
            break
        yview = log_text.yview()
        log_text.delete(0, END)
        working_count = 0
        zombie_count = 0
        sleeping_count = 0
        for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_times', 'memory_info', 'status']):
            try:
                pid = proc.info['pid']
                name = proc.info['name']
                username = proc.info['username']
                cpu_times = proc.info['cpu_times']
                memory_info = proc.info['memory_info']
                status = proc.info['status']

                user_time = cpu_times.user
                system_time = cpu_times.system

                label = (f"PID: {pid}, Name: {name}, User: {username}, "
                         f"User Time: {user_time}s, System Time: {system_time}s")

                log_message = (f"PID: {pid}, Name: {name}, User: {username}, "
                              f"User Time: {user_time}s, System Time: {system_time}s, "
                              f"Memory: {memory_info.rss / (1024 * 1024):.2f} MB, Status: {status}")

                logging.info(log_message)

                log_text.insert(END, label)
                log_text.see(END)

                # Count processes in working and sleeping states
                
                if status == psutil.STATUS_RUNNING:
                    working_count += 1
                elif status == psutil.STATUS_SLEEPING:
                    sleeping_count += 1
                elif status == psutil.STATUS_ZOMBIE:
                    zombie_count += 1

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Collect data for statistics
        current_time = time.strftime('%H:%M:%S')
        process_data['time'].append(current_time)
        process_data['working'].append(working_count)
        process_data['sleeping'].append(sleeping_count)
        process_data['zombie'].append(zombie_count)

        scrollbar.config(command=log_text.yview)
        time.sleep(5)

stop_flag = False

def wrapper():
    global stop_flag
    if start_button['text'] == "Запустить":
        stop_flag = False
        thread = threading.Thread(target=monitor_processes, daemon=True)
        thread.start()
        start_button.config(text="Остановить")
    elif(start_button['text'] == "Остановить"):
        stop_flag = True
        start_button.config(text="Запустить")

def plot_statistics():
    fig = Figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot(111)

    ax.plot(process_data['time'], process_data['working'], label='Working Processes')
    ax.plot(process_data['time'], process_data['sleeping'], label='Sleeping Processes')
    ax.plot(process_data['time'], process_data['zombie'], label='Zombie Processes')


    ax.set_xlabel('Time')
    ax.set_ylabel('Count of Processes')
    ax.set_title('Process Count Over Time')
    ax.legend()

    canvas = FigureCanvasTkAgg(fig, master=frame2)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    toolbar = NavigationToolbar2TkAgg(canvas, frame2)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

if __name__ == "__main__":
    root = tk.Tk()
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

    plot_button = ttk.Button(frame2, text="Показать статистику", command=plot_statistics)
    plot_button.pack(pady=10)

    root.mainloop()
