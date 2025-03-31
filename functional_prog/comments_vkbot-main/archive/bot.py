import logging
import json

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from vkbottle.api import API
from vkbottle.bot import Bot, Message, run_multibot
from vkbottle import GroupEventType, GroupTypes
from vkbottle.exception_factory import VKAPIError

from bot.HandleResponder import find_and_select_random_line_v2, process_text, find_file_in_directory


csv_dir_path = "./csv"
log_file_path = "./log.log"
tokens_file_path = "./tokens.json"


# enable logging
logging.basicConfig(filename=log_file_path, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# API for bot used in multibot is not required and may be
# forbidden later to avoid user-side mistakes with it's usage
# TIP: to access api in multibot handlers api should be
# requested from event.ctx_api
bot = Bot()

# Открываем файл с токенами
with open(tokens_file_path, "r") as file:
    data = json.load(file)

# Извлекаем токены из данных
tokens = data.get("tokens", [])

apis = [API(token) for token in tokens]

class TokensFileHandler(FileSystemEventHandler):
    def __init__(self, bot, tokens_file_path, apis_list):
        self.bot = bot
        self.tokens_file_path = tokens_file_path
        self.apis_list = apis_list

    def on_modified(self, event):
        if event.src_path == self.tokens_file_path:
            with open(self.tokens_file_path, "r") as file:
                data = json.load(file)
            tokens = data.get("tokens", [])
            self.apis_list.clear()
            self.apis_list.extend([API(token) for token in tokens])
            logging.info("Tokens updated")

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


# Read more about multibot in documentation
# high-level/bot/multibot
# run_multibot(bot, apis=apis)
if __name__ == "__main__":
    event_handler = TokensFileHandler(bot, tokens_file_path, apis)
    observer = Observer()
    observer.schedule(event_handler, path=tokens_file_path, recursive=False)
    observer.start()

    try:
        run_multibot(bot, apis=apis)
    finally:
        observer.stop()
        observer.join()
