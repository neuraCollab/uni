
import os
import json
import logging
import asyncio

from asyncio import get_event_loop, run_coroutine_threadsafe


from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from vkbottle.api import API
from vkbottle.bot import Bot, Message
from vkbottle import GroupEventType, GroupTypes
from vkbottle.exception_factory import VKAPIError

from HandleResponder import find_and_select_random_line_v2, process_text, find_file_in_directory

# Получение пути к текущему исполняемому файлу
current_file_path = os.path.abspath(__file__)

# Получение каталога из пути
current_directory = os.path.dirname(current_file_path)

os.chdir(current_directory)

csv_dir_path = "../csv"
log_file_path = "../.log"
tokens_file_path = "tokens.json"


# enable logging
logging.basicConfig(filename=log_file_path, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')


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
    
    # for massages in group
    @bot.on.message(lev="/инфо")  # lev > custom_rule from LevenshteinRule
    async def info(message: Message):
        current_group = (await message.ctx_api.groups.get_by_id())[0]
        await message.answer(f"Название моей группы: {current_group.name}")


    @bot.on.raw_event(GroupEventType.WALL_REPLY_NEW, dataclass=GroupTypes.WallReplyNew)
    async def on_wall_reply_new(event: GroupTypes.WallReplyNew):
        
        # print("\033[92m " + event.object.text)
        file_name=process_text(event.object.text + ".csv") # текст из поста
        found_file = find_file_in_directory(
            csv_dir_path, file_name
        )

        if found_file:
            logging.info(f"Файл {file_name} найден: {found_file}")
            response=find_and_select_random_line_v2(found_file, event.object.text) # текст из поста
            try:
                # Basic API call, please notice that bot.api is
                # not accessible in case multibot is used, API can be accessed from
                # event.ctx_api
                await event.ctx_api.wall.create_comment(
                    post_id=event.object.post_id,
                    owner_id=event.object.post_owner_id,
                    from_group=1,
                    message=response, # то что нужно отправить
                    reply_to_comment=event.object.id,
                )

            # Read more about exception handling in documentation
            # low-level/exception_handling/exception_handling
            except VKAPIError[901]:
                logging.error("Can't send message to user with id {}", event.object.user_id)
        else:
            logging.error(f"Файл {file_name} не найден в каталоге {csv_dir_path}.")

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

    # await asyncio.sleep(7)  # Пример ожидания
    # await bot_queue.put("vk1.a.gPC5q7YbPZK11ptAEQPk62TQM3AYM87RM8EH30Zq8j2XMesH7EnaZ57oBdSaPDfvL-XfkhW40FtvEhIgTNrrpkRiVj1KD9RWWLCHXLbbNLUDGHX3jt47VPCrferh1oWDOGlaoZ_TYyrLVl6QJkz-Y597u5Yp_qkQKewpmkZ5qqbiY5N-yex1ZV-MN2JnIU81F6KT3J666335xKUc0F60fg")

    await manager_task

if __name__ == "__main__":
    asyncio.run(main())