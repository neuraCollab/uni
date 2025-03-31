import subprocess

def find_usb_path_by_name(name):
    # Получение вывода команды lsblk
    lsblk_output = subprocess.check_output(["lsblk", "-o", "NAME,TYPE,MOUNTPOINT"]).decode("utf-8")

    # Разделение вывода на строки
    lines = lsblk_output.splitlines()

    # Поиск строки, содержащей заданное имя
    for line in lines[1:]:  # Пропустить заголовок
        parts = line.split()
        if len(parts) >= 3:
            device_name, device_type, mount_point = parts[:3]
            if device_name.startswith("/dev/") and device_type == "disk" and mount_point == name:
                return device_name

    return None

# Пример использования
flash_drive_name = "UBUNTU"
usb_path = find_usb_path_by_name(flash_drive_name)
if usb_path:
    print(f"Путь до флешки {flash_drive_name}: {usb_path}")
else:
    print(f"Флешка {flash_drive_name} не найдена.")
