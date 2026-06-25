import asyncio
from dotenv import load_dotenv
from pathlib import Path

from graph.graph import app

load_dotenv(Path(__file__).parent / "../.env")

async def main(query: str):
    return await app.ainvoke(input={"question": query})

if __name__ == "__main__":
    query = "What is agent memory?"
    res = asyncio.run(main(query))

    print(res["generation"])
