#!/bin/bash

current_program_name="nginx"

if [ -z "$your_domain" ]; then
  read -p "Введите ваш домен: " your_domain  # Попросить пользователя ввести значение для your_domain
  if [ -z "$your_domain" ]; then
    echo "Домен не был предоставлен. Выполнение скрипта остановлено."
    exit 1  # Остановить выполнение скрипта из-за отсутствия значения your_domain
  fi
fi

echo "Используемый домен: $your_domain"

# Let’s begin by updating the local package index to reflect the latest upstream changes:

sudo apt update

sudo apt install curl

# Then, install the nginx package:

sudo apt install nginx


# It is recommended that you enable the most restrictive profile that will still allow the traffic you’ve configured. Since we haven’t configured SSL for our server yet in this guide, we will only need to allow traffic on port 80:

sudo ufw allow 'Nginx HTTP'


# Check with the systemd init system to make sure the service is running by typing:

sudo systemctl status nginx

# To start the web server when it is stopped, type:

sudo systemctl start nginx



# -----Setting Up Virtual Hosts (Recommended)-----



# Create the directory for your_domain as follows:

sudo mkdir -p /var/www/"$your_domain"/html

# Next, assign ownership of the directory with the $USER environment variable:

sudo chown -R $USER:$USER /var/www/"$your_domain"/html

# The permissions of your web roots should be correct if you haven’t modified your umask value, which sets default file permissions. To ensure that your permissions are correct and allow the owner to read, write, and execute the files while granting only read and execute permissions to groups and others, you can input the following command:

sudo chmod -R 755 /var/www/"$your_domain"

# Next, create a sample index.html page using nano or your favorite editor:

cat > /var/www/"$your_domain"/html/index.html <<EOF
<html>
    <head>
        <title>Welcome to $your_domain!</title>
    </head>
    <body>
        <h1>Success! The $your_domain virtual host is working!</h1>
    </body>
</html>
EOF

# In order for nginx to serve this content, it’s necessary to create a virtual host file with the correct directives.

cat > /etc/nginx/sites-available/your_domain.conf <<EOF
server {
        listen 80;
        listen [::]:80;

        root /var/www/your_domain/html;
        index index.html index.htm index.nginx-debian.html;

        server_name your_domain www.your_domain;

        location / {
                try_files $uri $uri/ =404;
        }
}
EOF

# To avoid a possible hash bucket memory problem that can arise from adding additional server names

sudo ln -s /etc/nginx/sites-available/your_domain /etc/nginx/sites-enabled/

# find server_names_hash_bucket_size and change it to server_names_hash_bucket_size 64

sudo sed -i 's/server_names_hash_bucket_size .*/server_names_hash_bucket_size 64;/' /etc/nginx/nginx.conf

# Next, test to make sure that there are no syntax errors in any of your Nginx files:

sudo nginx -t

# If there aren’t any problems, restart Nginx to enable your changes:

sudo systemctl restart nginx

