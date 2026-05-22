from dotenv import load_dotenv
import os

load_dotenv()


def main():
    print("Hello from langraph-udemy-eden-marco!")
    print(os.environ.get("GROQ_API_KEY"))


if __name__ == "__main__":
    main()
