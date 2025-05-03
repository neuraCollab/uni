#! /bin/bash

# Check with the systemd init system to make sure the service is running by typing:

sudo systemctl status apache2

# To start the web server when it is stopped, type:

sudo systemctl start apache2



# -----Setting Up Virtual Hosts (Recommended)-----



# Create the directory for your_domain as follows:

sudo mkdir -p /var/www/"$your_domain"

# Next, assign ownership of the directory with the $USER environment variable:

sudo chown -R $USER:$USER /var/www/"$your_domain"

# The permissions of your web roots should be correct if you haven’t modified your umask value, which sets default file permissions. To ensure that your permissions are correct and allow the owner to read, write, and execute the files while granting only read and execute permissions to groups and others, you can input the following command:

sudo chmod -R 755 /var/www/"$your_domain"

# Next, create a sample index.html page using nano or your favorite editor:

cat > /var/www/"$your_domain"/index.html <<EOF
<html>
    <head>
        <title>Welcome to $your_domain!</title>
    </head>
    <body>
        <h1>Success! The $your_domain virtual host is working!</h1>
    </body>
</html>
EOF

# In order for Apache to serve this content, it’s necessary to create a virtual host file with the correct directives.

cat > /etc/apache2/sites-available/your_domain.conf <<EOF
<VirtualHost *:80>
    ServerAdmin webmaster@localhost
    ServerName your_domain
    ServerAlias www.your_domain
    DocumentRoot /var/www/your_domain
    ErrorLog \${APACHE_LOG_DIR}/error.log
    CustomLog \${APACHE_LOG_DIR}/access.log combined
</VirtualHost>
EOF

# Let’s enable the file with the a2ensite tool:

sudo a2ensite "$your_domain".conf

# Disable the default site defined in 000-default.conf:

sudo a2dissite 000-default.conf

# Next, let’s test for configuration errors:

sudo apache2ctl configtest

# Restart Apache to implement your changes:

sudo systemctl restart apache2


