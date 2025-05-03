import requests
from bs4 import BeautifulSoup

def get_latest_ubuntu_version():
    url = 'https://releases.ubuntu.com/'
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    version_tags = soup.find_all('a', class_='c-dropdown__list-link')

    if version_tags:
        latest_version_tag = version_tags[0]
        latest_version = latest_version_tag.text.strip()
        return latest_version

    return None

def download_latest_ubuntu_image(destination):
    latest_version = get_latest_ubuntu_version()

    if latest_version:
        url = f'https://releases.ubuntu.com/{latest_version}/ubuntu-{latest_version}-desktop-amd64.iso'

        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(destination, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        print(f"Загрузка последнего стабильного образа Ubuntu ({latest_version}) завершена. Файл сохранен как {destination}.")
    else:
        print("Не удалось получить информацию о последней версии Ubuntu.")

# Пример использования функции для скачивания последнего стабильного образа Ubuntu Desktop Edition.
destination = 'ubuntu-desktop.iso'

download_latest_ubuntu_image(destination)
