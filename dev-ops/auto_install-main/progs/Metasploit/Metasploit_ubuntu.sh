#!/bin/bash

current_programm_name="metasploit"

# В начале необходимо установить Metasploit Framework. Для этого поставим все необходимые пакеты. 
sudo apt-get install ruby libopenssl-ruby libyaml-ruby libdl-ruby libiconv-ruby libreadline-ruby irb ri rubygems
sudo apt-get install subversion
sudo apt-get build-dep ruby
sudo apt-get install ruby-dev libpcap-dev
sudo apt-get install rubygems libsqlite3-dev
sudo gem install sqlite3-ruby
sudo apt-get install rubygems libmysqlclient-dev
sudo gem install mysql

# Далее скачиваем Metasploit Framework вот тут: http://downloads.metasploit.com/data/releases/metasploit-latest-linux-installer.run:

wget http://downloads.metasploit.com/data/releases/metasploit-latest-linux-installer.run
chmod +x metasploit-latest-linux-installer.run

#  И запускаем…

sudo ./metasploit-latest-linux-installer.run

# После установки необходимо обновить Metasploit Framework

sudo msfupdate


