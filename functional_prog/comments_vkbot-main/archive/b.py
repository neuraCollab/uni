import logging
import json
import asyncio
from vkbottle.bot import Bot, Message
from vkbottle.api import API
from vkbottle import GroupEventType, GroupTypes
from vkbottle.exception_factory import VKAPIError
from bot.HandleResponder import find_and_select_random_line_v2, process_text, find_file_in_directory
import nest_asyncio

nest_asyncio.apply()

csv_dir_path = "./csv"
log_file_path = "./log.log"
tokens_file_path = "./tokens.json"

logging.basicConfig(filename=log_file_path, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_tokens():
    with open(tokens_file_path, "r") as file:
        return json.load(file)['tokens']

async def bot_instance(token):
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

    await bot.run_polling()

async def main():
    tokens = load_tokens()
    # await bot_instance(tokens[0])
    [await bot_instance(token) for token in tokens]
    # tasks = [asyncio.create_task(bot_instance(token)) for token in tokens]
    # await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
