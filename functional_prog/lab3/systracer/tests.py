import psutil

def monitor_processes():
    # Iterate over all running processes
    for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_info']):
        try:
            # Get process details
            pid = proc.info['pid']
            name = proc.info['name']
            username = proc.info['username']
            cpu_percent = proc.info['cpu_percent']
            memory_info = proc.info['memory_info']

            # Print process details
            print(f"PID: {pid}, Name: {name}, User: {username}, CPU: {cpu_percent}%, Memory: {memory_info.rss / (1024 * 1024):.2f} MB")

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Handle exceptions for processes that may have terminated or are inaccessible
            pass

if __name__ == "__main__":
    monitor_processes()
