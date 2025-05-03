# vk_bots


chmod 777 ./tokens.json

cd /etc/systemd/system/

sudo systemctl daemon-reload

sudo systemctl enable mybot.service

sudo systemctl start mybot.service

sudo systemctl status mybot.service

export FASTAPI_ENV=production

gunicorn -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:443 --certfile=server.crt --keyfile=server.key -D main:app

# Для остановки
pkill -f gunicorn
