#!/bin/bash

current_programm_name=\"$directory_name\"

# Add the Wireshark and TShark repository:

sudo add-apt-repository -y ppa:wireshark-dev/stable

# Install TShark:

sudo apt install -y tshark

# Run the following command to add the current user to a wireshark group:

sudo usermod -a -G wireshark $USER
