from telethon import TelegramClient
from telethon.errors.rpcerrorlist import PeerIdInvalidError

# Вставьте ваши данные
api_id = 24791827
api_hash = 'd8ff802fab2f32be99b7cf9b0e653ffa'
message_template = """{name}, привет! 🙌🏼 
Меня зовут Mike, я менеджер международной организации AIESEC.

Вчера ты был(а) на Management Career Week и оставил(а) свой контакт. Как раз сейчас у нас открыт набор на профессиональные и преподавательские стажировки за границей 🌍
Доступно много направлений в разных странах — есть из чего выбрать.

Подскажи, будет ли тебе удобно созвониться сегодня в 20:00 МСК, чтобы я мог рассказать о возможностях подробнее?)"""

# Список телеграм-ников
usernames = [
    '@aqawue',
    '@belanova_tanya',
    '@alex_or_not',
    '@id_szu',
    '@Telosk',
    '@Jasur',
    '@dariner',
    '@sofilifelove'
]

# Запуск клиента
client = TelegramClient('aiesec_session', api_id, api_hash)

async def main():
    for username in usernames:
        try:
            name = username.strip('@').split()[0].capitalize()
            user = await client.get_entity(username.strip())
            await client.send_message(user, message_template.format(name=name))
            print(f"✅ Отправлено: {username}")
        except PeerIdInvalidError:
            print(f"❌ Не удалось найти пользователя: {username}")
        except Exception as e:
            print(f"⚠️ Ошибка с {username}: {e}")

with client:
    client.loop.run_until_complete(main())
