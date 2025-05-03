import json

import os

from vkbottle.bot import Bot
from vkbottle import GroupEventType, GroupTypes
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import asyncio
from asyncio import get_event_loop, run_coroutine_threadsafe

class TokenFileEventHandler(FileSystemEventHandler):
    def __init__(self, filename, bot_queue):
        self.filename = filename
        self.bot_queue = bot_queue
        self.known_tokens = self.load_initial_tokens()
        self.loop = get_event_loop()  # Сохраняем ссылку на цикл событий

    def load_initial_tokens(self):
        with open(self.filename, "r") as file:
            tokens = set(json.load(file))
            return tokens

    def on_modified(self, event):
        if os.path.abspath(event.src_path) == os.path.abspath(self.filename):
            with open(self.filename, "r") as file:
                tokens = set(json.load(file))
                new_tokens = tokens - self.known_tokens
                for token in new_tokens:
                    # Запускаем асинхронную корутину в цикле событий
                    run_coroutine_threadsafe(self.bot_queue.put(token), self.loop)
                self.known_tokens = tokens

async def run_bot(token):
    bot = Bot(token)

    @bot.on.raw_event(GroupEventType.WALL_REPLY_NEW, dataclass=GroupTypes.WallReplyNew)
    async def comment_handler(event: GroupTypes.WallReplyNew):
        # Обработчик событий
        pass
    
    

    await bot.run_polling()

async def bot_manager(bot_queue):
    while True:
        token = await bot_queue.get()
        asyncio.create_task(run_bot(token))

async def main():
    bot_queue = asyncio.Queue()
    manager_task = asyncio.create_task(bot_manager(bot_queue))

    event_handler = TokenFileEventHandler("tokens.json", bot_queue)
    observer = Observer()
    observer.schedule(event_handler, path=".", recursive=False)
    observer.start()

    # Загрузка и запуск начальных ботов
    initial_tokens = event_handler.load_initial_tokens()
    for token in initial_tokens:
        await bot_queue.put(token)

    await manager_task

if __name__ == "__main__":
    asyncio.run(main())