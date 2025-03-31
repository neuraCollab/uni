import os
import json
import logging
import asyncio

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


async def run_bot(token):
    bot = Bot(token)

    @bot.on.raw_event(GroupEventType.WALL_REPLY_NEW, dataclass=GroupTypes.WallReplyNew)
    async def comment_handler(event: GroupTypes.WallReplyNew):
        # Обработчик событий
        pass
    
    # for massages in group
    @bot.on.message(lev="/инфо")  # lev > custom_rule from LevenshteinRule
    async def info(message: Message):
        current_group = (await message.ctx_api.groups.get_by_id())[0]
        await message.answer(f"Название моей группы: {current_group.name}")


    #TODO чекнуть кому какой текст принадлежит

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

    # Загрузка токенов из файла
    with open(tokens_file_path, "r") as file:
        tokens = json.load(file)

    for token in tokens:
        await bot_queue.put(token)
    
    await asyncio.sleep(10)  # Пример ожидания
    await bot_queue.put("vk1.a.fZQeEp_APlqUFXPam-MIme_WzzqKkHyH1xLOj3H7Wd3v7tBKsgb4Gn3dnC0nGKQQE2JvpKIBW2JCcMmhCHo4xOJpwRk2Jnk4tqkW8IJsjuDdefJSlwNsVisabAq51Cw4K2D4MCAVMrOEtm4LaYK7tfytnTeqXEj-pU1cbCgo7M8-uJIHN4pAPGYjyW4HFzHX_l0VCMCwkxNj87GCyRmPHA")

    await manager_task

if __name__ == "__main__":
    asyncio.run(main())
