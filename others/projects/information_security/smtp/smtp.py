# send_attachment.py
# import necessary packages
import os
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
import smtplib

# create message object instance
msg = MIMEMultipart()


message = "Thank you"

# setup the parameters of the message
password = "JavaScript12"
msg['From'] = "mihailsarabarin00@gmail.com"
msg['To'] = "Zelendra12@mail.ru"
msg['Subject'] = "Photos"


file = open("1.png", 'rb').read()
image = MIMEImage(file, name=os.path.basename("1.png"))

# add in the message body
msg.attach(MIMEText(message, 'plain'))
msg.attach(image)

#create server
server = smtplib.SMTP('smtp.gmail.com: 587')


server.starttls()

# Login Credentials for sending the mail
server.login(msg['From'], password)


# send the message via the server.
server.sendmail(msg['From'], msg['To'], msg.as_string())

server.quit()

print ("successfully sent email to %s:" % (msg['To']))
