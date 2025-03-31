#!/bin/bash

current_programm_name="docker compose"


# Check docker_engine install
echo "Checking docker-engine instal..."

if check_install "docker"; then
    install_script "../dokcer_engine/docker_engine_ubuntu.sh"
fi

# For non-Gnome Desktop environments, gnome-terminal must be installed:
sudo apt install gnome-terminal

# Uninstall the tech preview or beta version of Docker Desktop for Linux. Run:
sudo apt remove docker-desktop

# For a complete cleanup, remove configuration and data files at $HOME/.docker/desktop,
#  the symlink at /usr/local/bin/com.docker.cli,
#  and purge the remaining systemd service files.
rm -r $HOME/.docker/desktop
sudo rm /usr/local/bin/com.docker.cli
sudo apt purge docker-desktop

# install pup for parsing
sudo wget https://github.com/ericchiang/pup/releases/download/v0.4.0/pup_v0.4.0_linux_amd64.zip 
sudo unzip -o ./pup_v0.4.0_linux_amd64.zip -d /usr/local/bin 
rm ./pup_v0.4.0_linux_amd64.zip

# --- Install lastest .deb from ---

# parse download link
URL="https://docs.docker.com/desktop/install/ubuntu/"
docker_desktop_link = $(wget -O- "$URL" | pup ' article ol li:nth-of-type(2) p a attr{href}')

# Get name of downloaded file
fileWithParams=$(basename "$docker_desktop_link")
fileName=${fileWithParams%%\?*}
docker_desktop_package_name=${file%.deb}
docker_desktop_deb=$(find ~ -type f -name "*docker-desktop-4.25.1-amd64*" | head -n 1)

# Install
sudo apt-get update
sudo apt-get install $docker_desktop_deb

# Start docker desktop
systemctl --user start docker-desktop

# check_installation 
echo "Don't forget to install gpg key for sign in. 
    More information: https://docs.docker.com/desktop/get-started/"