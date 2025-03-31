import os
import subprocess

def install_balena_etcher():
    # Установка зависимостей
    subprocess.call(["sudo", "apt", "install", "apt-transport-https", "curl", "-y"])

    # Добавление ключа GPG репозитория balenaEtcher
    subprocess.call(["curl", "-1sLf", "https://balena.io/etcher/static/etcher_repo.key", "|", "sudo", "-E", "apt-key", "add", "-"])

    # Добавление репозитория balenaEtcher в систему
    subprocess.call(["echo", "deb", "https://deb.etcher.io stable etcher", "|", "sudo", "tee", "/etc/apt/sources.list.d/balena-etcher.list"])

    # Обновление списка пакетов
    subprocess.call(["sudo", "apt", "update"])

    # Установка balenaEtcher
    subprocess.call(["sudo", "apt", "install", "balena-etcher-electron", "-y"])

def run_balena_etcher():
    # Запуск balenaEtcher
    subprocess.call(["balena-etcher-electron"])

# Пример использования
install_balena_etcher()
run_balena_etcher()
