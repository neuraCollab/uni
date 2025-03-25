from scapy.all import sniff, IP, ICMP, TCP, UDP
from scapy.all import *
import tkinter as tk
from tkinter import ttk, messagebox, Listbox, END
import threading


sniffing_flag = False

blocked_ips = []
detected_ips = []
all_ips = []

def is_detected_ip(packet):
    banned_port = [10, 11, 22, 421]

    if len(packet) > 10:
        return True

    if packet.haslayer(TCP) and packet[TCP].flags == 'S':
        return True
    
    if packet.haslayer(TCP) and (packet[TCP].dport in banned_port or packet[TCP].sport in banned_port):
        return True
    
    if packet.haslayer(UDP) and (packet[UDP].dport in banned_port or packet[UDP].sport in banned_port):
        return True

    if packet.haslayer(TCP) and packet[TCP].flags == 'A':
        return False

    return False

def packet_callback(packet):
    all_ips.append(packet.summary())
    update_all_ip()

    if not is_detected_ip(packet):
        # print("true packet")
        if IP in packet:
            ip_layer = packet.getlayer(IP)
            # print(packet.summary())
            packet = IP(dst=ip_layer.src) / ICMP()
    else:
        if packet[IP].src not in detected_ips:
            detected_ips.append(packet[IP].src)
        update_detected_ip()


def block_packet(packet):
    ip = IP(dst=packet[IP].src, src=packet[IP].dst)
    icmp = ICMP(type=3, code=3)
    send(ip/icmp/packet[IP])

def update_all_ip():
    yview = all_ip.yview()
    all_ip.delete(0, END)
    for packet in all_ips:
        all_ip.insert(END, packet)
    if yview[1] == 1.0:
        all_ip.see(END)


def update_banned_ip():
    yview = banned_ip.yview()
    banned_ip.delete(0, END)
    for ip in blocked_ips:
        banned_ip.insert(END, ip)
    if yview[1] == 1.0:
        banned_ip.see(END)


def update_detected_ip():
    yview = detected_ip.yview()
    detected_ip.delete(0, END)
    for ip in detected_ips:
        detected_ip.insert(END, ip)
    if yview[1] == 1.0:
        detected_ip.see(END)


def start_sniffer():
    sniff(prn=packet_callback, store=False, stop_filter=lambda x: not sniffing_flag)

def wrapper(start_button):
    global sniffing_flag
    if(sniffing_flag == False):
        sniffing_flag = True
        thread = threading.Thread(target = start_sniffer, daemon=True)
        thread.start()
        start_button.config(text="Остановить")
    else:
        sniffing_flag = False
        start_button.config(text="Запуск")


def block_ip():
    try:
        selected_ip = detected_ip.get(detected_ip.curselection())
        if selected_ip:
            if selected_ip not in blocked_ips:
                blocked_ips.append(selected_ip)
                command = f"sudo iptables -A INPUT -s {selected_ip} -j DROP"
                os.system(command)
                update_banned_ip()
                messagebox.showinfo("IP Заблокирован", f"IP {selected_ip} был успешно заблокирован.")
            else:
                messagebox.showwarning("Ошибка", "IP уже заблокирован.")
    except:
        messagebox.showwarning("Ошибка", "Ошибка блокировки.")

def unblock_ip():
    try:
        selected_ip = banned_ip.get(banned_ip.curselection())
        if selected_ip:
            blocked_ips.remove(selected_ip)
            command = f"sudo iptables -D INPUT -s {selected_ip} -j DROP"
            os.system(command)
            print(selected_ip)
            update_banned_ip()
            messagebox.showinfo("IP Разблокирован", f"IP {selected_ip} был успешно разблокирован.")
    except:
        messagebox.showwarning("Ошибка", "Ошибка разблокировки.")



if __name__ == "__main__":    
    root = tk.Tk()
    root.title("Сетевой трафик")
    root.geometry("800x700")

    left_frame = ttk.Frame(root)
    left_frame.grid(column=0, row=0, padx = 10, pady = 10, ipadx=6, ipady=6)


    banned_label = ttk.Label(left_frame, text="Получаемые пакеты")
    banned_label.grid(column=0, row=0)
    all_ip = Listbox(left_frame, width=30, height=30)
    all_ip.grid(column=0, row=1) 
    start_button = ttk.Button(left_frame, text="Запуск", command=lambda: wrapper(start_button))
    start_button.grid(column=0, row=2)


    middle_frame = ttk.Frame(root)
    middle_frame.grid(column=1, row=0, padx = 10, pady = 10, ipadx=6, ipady=6)


    detected_label = ttk.Label(middle_frame, text="Подозрительные IPs")
    detected_ip = Listbox(middle_frame, width=30, height=30)
    detected_label.grid(column=0, row=0)
    detected_ip.grid(column=0, row=1)
    banned_button = ttk.Button(middle_frame, text="Заблокировать", command=block_ip)
    banned_button.grid(column=0, row=2)


    right_frame = ttk.Frame(root)
    right_frame.grid(column=2, row=0, padx = 10, pady = 10, ipadx=6, ipady=6,)


    banned_label = ttk.Label(right_frame, text="Заблокированные IPs")
    banned_label.grid(column=0, row=0)
    banned_ip = Listbox(right_frame, width=30, height=30)
    banned_ip.grid(column=0, row=1)
    unbanned_button = ttk.Button(right_frame, text="Разблокировать", command=unblock_ip)
    unbanned_button.grid(column=0, row=2), 

    root.mainloop()
