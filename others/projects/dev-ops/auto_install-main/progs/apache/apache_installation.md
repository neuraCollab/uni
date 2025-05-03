# apache instalation

> sourse: https://www.digitalocean.com/community/tutorials/how-to-install-the-apache-web-server-on-ubuntu-20-04

## after instalation You can test this by navigating to http://your_domain

List the ufw application profiles by typing:

`sudo ufw app list`

You can verify the change by typing:

`sudo ufw status`

To re-enable the service to start up at boot, type:

`sudo systemctl enable apache2`

Use curl to retrieve icanhazip.com using IPv4:

`curl -4 icanhazip.com`

