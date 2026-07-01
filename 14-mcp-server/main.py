import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / "../.env")

async def main():
    print("MCP Server")

if __name__ == "__main__":
    asyncio.run(main())
