import subprocess

def install_ubuntu_image(image_path, flash_drive_path):
    # Запуск BalenaEtcher из командной строки
    subprocess.run(['balena-etcher', '--flash', image_path, '--drive', flash_drive_path])

# Пример использования
ubuntu_image_path = '/path/to/ubuntu.iso'  # Укажите путь до образа Ubuntu
flash_drive_path = '/dev/sdX'  # Укажите путь до флешки (замените 'sdX' на фактический путь)

install_ubuntu_image(ubuntu_image_path, flash_drive_path)
