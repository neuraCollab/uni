# tshark instalation

> how to use: https://selectel.ru/blog/analiz-setevogo-trafika-na-servere-pri-pomoshhi-tshark/

To make changes to take effect, logout and login to your machine. After reconnection, you can check TShark version:

`tshark --version`

Execute tshark command without any arguments to start capturing packets on default network interface:

`tshark`

We can find network interfaces which are available to the TShark with command:

`tshark -D`

The -i option allows capturing packets on a specific network interface.

`tshark -i ens33`

Uninstall TShark

If you wish to completely remove TShark and all related dependencies, execute the following command:

`sudo apt purge --autoremove -y tshark`

Remove GPG key and repository:

`sudo rm -rf /etc/apt/trusted.gpg.d/wireshark-dev-ubuntu-stable.gpg*`

`sudo rm -rf /etc/apt/sources.list.d/wireshark-dev-ubuntu-stable-jammy.list`