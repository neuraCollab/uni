import asyncio
import re

ALL_GROUPS = set()

async def write_message(writer, data):
    writer.write(data)
    await writer.drain()

async def connect_user(reader, writer):
    global ALL_USERS, ALL_GROUPS

    name_bytes = await reader.read(100)
    data = name_bytes.decode().strip().split()
    name = data[0]

    if(name in ALL_USERS.keys()):
        writer.write("SIG1".encode())
        await writer.drain()
        return
    else:
        writer.write("SIG2".encode())
        await writer.drain()

    group = data[1]

    ALL_GROUPS.add(group)

    ALL_USERS[name] = (reader, writer, group)

    await broadcast_message(f'{name} has connected', group)

    welcome = f'Welcome {name}. If you need help, please, write @help or write @list if you want to know exists groups.'
    writer.write(welcome.encode())
    await writer.drain()


    return name, group

async def handle_chat_client(reader, writer):
    try:
        name, group = await connect_user(reader, writer)
        while True:
            data = await reader.read(100)
            message = data.decode()

            if message == "QUIT":
                break

            addr = writer.get_extra_info('peername')
            print(f"Received {message!r} from {addr!r}")

            message = f"[{name}] {message}"

            await broadcast_message(message, group)
    finally:
        await disconnect_user(name, group, writer)

async def broadcaster():
    global queue, ALL_USERS

    while True:
        packet = await queue.get()
        message = packet[0]

        private_users = re.findall(r'\[(.*?)\]', message)
        
        author = packet[1]

        request_by_server = re.findall(r'@....', message)

        
        if(len(request_by_server) > 0 and request_by_server[0] == '@help'):
            message = "[Server]: Вы можете отправлять личные сообщения через [имя]"
        elif(len(request_by_server) > 0 and request_by_server[0] == '@list'):
            message = "[Server]: Groups: " + " ".join(list(ALL_GROUPS))

        msg_bytes = message.encode()

        tasks = []
        print(private_users)
        if request_by_server:
            tasks = [asyncio.create_task(write_message(w, msg_bytes)) if(room == author and user == private_user) else asyncio.create_task(asyncio.sleep(0)) for user,(_,w, room) in ALL_USERS.items() for private_user in private_users]
        elif(len(private_users) > 1):
            tasks = [asyncio.create_task(write_message(w, msg_bytes)) if(room == author and user == private_user) else asyncio.create_task(asyncio.sleep(0)) for user,(_,w, room) in ALL_USERS.items() for private_user in private_users]
        else:
            tasks = [asyncio.create_task(write_message(w, msg_bytes)) if(room == author) else asyncio.create_task(asyncio.sleep(0)) for _,(_,w, room) in ALL_USERS.items()]
        _ = await asyncio.wait(tasks)

async def broadcast_message(message, author):
    global queue
    await queue.put((message, author))

async def disconnect_user(name, group, writer):
    global ALL_USERS

    writer.close()
    await writer.wait_closed()

    del ALL_USERS[name]
    
    await broadcast_message(f'{name} has disconnected', group)

async def main():
    broadcaster_task = asyncio.create_task(broadcaster())
    ip_address =  '127.0.0.2'
    port = 8888
    server = await asyncio.start_server(
        handle_chat_client, ip_address, port)

    addr = server.sockets[0].getsockname()
    print(f'Serving on {addr}')

    async with server:
        await server.serve_forever()

ALL_USERS = {}

queue = asyncio.Queue()

asyncio.run(main())