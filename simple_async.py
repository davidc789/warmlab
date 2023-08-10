import asyncio
import aiosqlite
import contextlib


async def coroutine():
    async with contextlib.AsyncExitStack() as stack:
        conn = await aiosqlite.connect("data.sqlite")
        await conn.execute("PRAGMA QUERY_ONLY = FALSE")
        print("Hello world!")
        await conn.close()


if __name__ == '__main__':
    asyncio.run(coroutine())
