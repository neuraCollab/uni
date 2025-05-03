import logging
import json
import asyncio
from vkbottle.bot import Bot, Message
from vkbottle.api import API
from vkbottle import GroupEventType, GroupTypes
from vkbottle.exception_factory import VKAPIError
from bot.HandleResponder import find_and_select_random_line_v2, process_text, find_file_in_directory
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import nest_asyncio

nest_asyncio.apply()

csv_dir_path = "./csv"
log_file_path = "./log.log"
tokens_file_path = "./tokens.json"

logging.basicConfig(filename=log_file_path, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_tokens():
    with open(tokens_file_path, "r") as file:
        return json.load(file)['tokens']
    
# Словарь для хранения активных задач ботов
active_bots = {}

async def start_bot(token):
    bot = Bot(token=token)

    @bot.on.message(text='/инфо')
    async def info_handler(message: Message):
        current_group = (await message.ctx_api.groups.get_by_id())[0]
        await message.answer(f'Название моей группы: {current_group.name}')

    @bot.on.raw_event(GroupEventType.WALL_REPLY_NEW, dataclass=GroupTypes.WallReplyNew)
    async def wall_reply_handler(event: GroupTypes.WallReplyNew):
        file_name = process_text(event.object.text + ".csv")
        found_file = find_file_in_directory(csv_dir_path, file_name)
        if found_file:
            response = find_and_select_random_line_v2(found_file, event.object.text)
            try:
                await event.ctx_api.wall.create_comment(
                    post_id=event.object.post_id,
                    owner_id=event.object.post_owner_id,
                    from_group=1,
                    message=response,
                    reply_to_comment=event.object.id,
                )
            except VKAPIError[901]:
                logging.error("Can't send message to user with id {}", event.object.user_id)

    active_bots[token] = bot
    await bot.run_forever()
    

# Функция для остановки бота
async def stop_bot(token):
    if token in active_bots:
        await active_bots[token].http.close()
        del active_bots[token]

# Обработчик изменений файла
class TokenFileChangeHandler(FileSystemEventHandler):
    def __init__(self, loop):
        self.loop = loop
        self.current_tokens = set()

    def on_modified(self, event):
        if event.src_path == tokens_file_path:
            asyncio.run_coroutine_threadsafe(self.async_on_modified(), self.loop)

    async def async_on_modified(self):
        new_tokens = set(load_tokens())
        added_tokens = new_tokens - self.current_tokens
        removed_tokens = self.current_tokens - new_tokens

        for token in added_tokens:
            asyncio.create_task(start_bot(token))
        
        for token in removed_tokens:
            asyncio.create_task(stop_bot(token))

        self.current_tokens = new_tokens

class BotManager:
    def __init__(self):
        self.bots = {}
        self.running = False

    async def start_bot(self, token):
        if token in self.bots:
            return  # Bot уже запущен

        bot = Bot(token=token)
        self.bots[token] = bot
        asyncio.create_task(self.run_bot(bot))

    async def run_bot(self, bot):
        await bot.run_polling()  # Запускаем бота

    async def stop_bot(self, token):
        if token in self.bots:
            bot = self.bots.pop(token)
            
                
# async def main():
    # loop = asyncio.get_running_loop()
    # file_change_handler = TokenFileChangeHandler(loop)
    # observer = Observer()
    # observer.schedule(file_change_handler, tokens_file_path, recursive=False)
    # observer.start()

    # current_tokens = set(load_tokens())
    # file_change_handler.current_tokens = current_tokens
    # for token in current_tokens:
    #     asyncio.create_task(start_bot(token))

    # await asyncio.Event().wait()  # Бесконечное ожидание для предотвращения завершения скрипта

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    file_change_handler = TokenFileChangeHandler(loop)
    observer = Observer()
    observer.schedule(file_change_handler, tokens_file_path, recursive=False)
    observer.start()

    current_tokens = set(load_tokens())
    file_change_handler.current_tokens = current_tokens
    for token in current_tokens:
        loop.create_task(start_bot(token))

    try:
        loop.run_forever()
    finally:
        observer.stop()
        observer.join()