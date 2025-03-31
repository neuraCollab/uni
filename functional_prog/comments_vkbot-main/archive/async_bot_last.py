import asyncio
from vkbottle.bot import Bot, Message

# Функция для инициализации бота с заданным токеном
async def create_and_run_bot(token):
    bot = Bot(token)

    # Настройка обработчиков событий
    @bot.on.message(text=["привет", "hello"])
    async def greeting_handler(message: Message):
        await message.answer("Привет! Я бот на VKBottle.")

    # Запуск бота
    await bot.run_polling()

# Главная функция для запуска всех ботов
async def run_bots(tokens):
    tasks = [create_and_run_bot(token) for token in tokens]
    await asyncio.gather(*tasks)

# Список токенов для ботов
tokens = ["token1", "token2", "token3"]

# Запуск всех ботов асинхронно
asyncio.run(run_bots(tokens))
