"""Async chat server/client demo - core asyncio interview patterns.

Anchor example for ../concurrency-async.md (asyncio section). Demonstrates:
  - asyncio.start_server / asyncio.open_connection (event-driven TCP I/O)
  - asyncio.Queue used as a single-consumer "broadcaster" channel
  - asyncio.create_task for fire-and-forget concurrent work
  - asyncio.gather to fan a message out to many clients concurrently
  - graceful disconnect handling with try/finally

Run in two separate terminals:
    python async_chat_server.py server
    python async_chat_server.py client alice room1
    python async_chat_server.py client bob room1

Adapted from a university networking lab (originally split across a
Tkinter-GUI client and a server script). The GUI layer has been removed
entirely, both roles were merged into one file, and type hints plus a
`ChatServer` class were added to make the state ownership explicit.
"""

from __future__ import annotations

import asyncio
import sys
from asyncio import StreamReader, StreamWriter
from dataclasses import dataclass

HOST = "127.0.0.1"
PORT = 8888


@dataclass
class Client:
    name: str
    room: str
    writer: StreamWriter


class ChatServer:
    """Owns all shared, mutable server state.

    Keeping state on an instance (instead of module-level globals, as in the
    original lab code) makes the concurrency story explicit: every coroutine
    below is a method that shares `self.clients` and `self.queue`, and only
    `broadcaster()` ever writes to a socket - that's what avoids interleaved
    writes when many clients are active at once.
    """

    def __init__(self) -> None:
        self.clients: dict[str, Client] = {}
        # Producer/consumer channel: handlers publish, one task consumes.
        self.queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()  # (message, room)

    async def broadcaster(self) -> None:
        """The single consumer task: drains the queue, fans out to a room.

        Centralizing all outbound writes in one task is the key trick - it
        means N client-handler coroutines can call `publish()` concurrently
        without ever racing on a socket write.
        """
        while True:
            message, room = await self.queue.get()
            targets = [c.writer for c in self.clients.values() if c.room == room]
            if targets:
                await asyncio.gather(*(self._send(w, message) for w in targets))

    @staticmethod
    async def _send(writer: StreamWriter, message: str) -> None:
        writer.write(message.encode())
        await writer.drain()

    async def publish(self, message: str, room: str) -> None:
        await self.queue.put((message, room))

    async def handle_client(self, reader: StreamReader, writer: StreamWriter) -> None:
        """Per-connection coroutine. asyncio.start_server spawns one of these
        as a Task for every accepted connection - that's the concurrency."""
        name, room = await self._register(reader, writer)
        if name is None:
            return
        try:
            while True:
                data = await reader.read(1024)
                if not data:
                    break
                text = data.decode().strip()
                if text == "QUIT":
                    break
                await self.publish(f"[{name}] {text}", room)
        finally:
            await self._unregister(name, room)

    async def _register(
        self, reader: StreamReader, writer: StreamWriter
    ) -> tuple[str | None, str | None]:
        raw = await reader.read(100)
        try:
            name, room = raw.decode().strip().split(maxsplit=1)
        except ValueError:
            writer.write(b"ERR:bad_handshake")
            await writer.drain()
            return None, None

        if name in self.clients:
            writer.write(b"ERR:name_taken")
            await writer.drain()
            return None, None

        self.clients[name] = Client(name, room, writer)
        await self.publish(f"{name} has joined", room)
        writer.write(f"Welcome {name}!".encode())
        await writer.drain()
        return name, room

    async def _unregister(self, name: str | None, room: str | None) -> None:
        if name is None:
            return
        client = self.clients.pop(name, None)
        if client is None:
            return
        client.writer.close()
        await client.writer.wait_closed()
        await self.publish(f"{name} has left", room)


async def run_server() -> None:
    state = ChatServer()
    broadcaster_task = asyncio.create_task(state.broadcaster())
    server = await asyncio.start_server(state.handle_client, HOST, PORT)
    addr = server.sockets[0].getsockname()
    print(f"Serving on {addr}")
    try:
        async with server:
            await server.serve_forever()
    finally:
        broadcaster_task.cancel()


async def run_client(name: str, room: str) -> None:
    reader, writer = await asyncio.open_connection(HOST, PORT)
    writer.write(f"{name} {room}".encode())
    await writer.drain()

    async def listen() -> None:
        while True:
            data = await reader.read(1024)
            if not data:
                break
            print(data.decode())

    listener = asyncio.create_task(listen())
    loop = asyncio.get_event_loop()
    try:
        while True:
            # run_in_executor keeps blocking input() from freezing the event loop.
            line = await loop.run_in_executor(None, input)
            writer.write(line.encode())
            await writer.drain()
            if line == "QUIT":
                break
    finally:
        listener.cancel()
        writer.close()
        await writer.wait_closed()


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return
    if sys.argv[1] == "server":
        asyncio.run(run_server())
    elif sys.argv[1] == "client" and len(sys.argv) >= 4:
        asyncio.run(run_client(sys.argv[2], sys.argv[3]))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
