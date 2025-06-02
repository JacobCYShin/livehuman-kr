import asyncio

async def task(name):
    await asyncio.sleep(1)
    print(f"{name} 끝!")

async def main():
    await asyncio.gather(
        task("A"),
        task("B")
    )

asyncio.run(main())
