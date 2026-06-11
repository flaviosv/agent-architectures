import os
from pathlib import Path
from dotenv import load_dotenv
import ollama
from langsmith import traceable

load_dotenv(Path(__file__).parent / "../.env")

MAX_ITERATIONS = 10
MODEL = os.getenv("OLLAMA_MODEL")

# It's not necessary write doen all those details, if the string
# has docsgring in Google docstring format,, and lass the functions lile
# tooks_for_llm = [get_product_price], it's gonna work the same
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of product in the catalog",
            "parameters": {
                "type": "object",
                "required": ["product"],
                "properties": {
                    "product": {"type": "string", "description": "Product name"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount to the price based on the customer's tier. Available tiers: bronze, silver, gold",
            "parameters": {
                "type": "object",
                "required": ["price", "discount_tier"],
                "properties": {
                    "price": {"type": "float", "description": "Product's price"},
                    "discount_tier": {"type": "string", "description": "Discount tier, available options: bronze, silver, gold"}
                }
            }
        }
    }
]

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of product in the catalog."""
    print(f"Executing get_product_price({product})")
    prices = {"laptop": 1300, "headphones": 150, "keyboard": 90}

    return prices.get(product, 0)

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount to the price based on the customer's tier.
    Available tiers: bronze, silver, gold
    """
    print(f"Executing apply_discount({price}, {discount_tier})")
    percentages = {"bronze": 5, "silver": 12, "gold": 23}

    return price * (1 - percentages.get(discount_tier, 0) / 100)


traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)

@traceable(name="Langchain Agent Loop")
def main(prompt: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount
    }

    print(f"Question: {prompt}")
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant\n"
                "You have access to a product catalog tool"
                "and a discoint tool\n\n"
                "STRICT RULES - yoh must follow these exactly: \n"
                "1. NEVER guess or assume any product price"
                "2. Only call apply_discount AFTER you've received \n"
                "a price from get_product_price. pass the exact price \n"
                "returned by get_product_price - do OT pass a made-up number.\n"
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    for interation in range(1, MAX_ITERATIONS + 1):
        print(f"--- ITERATION {interation} ---")
        response = ollama_chat_traced(messages)
        ai_message = response.message

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"Final answer: {ai_message.content}")
            return ai_message.content

        messages.append({
            "role": "assistant",
            "content": str(ai_message)
        })

        for tool_c in tool_calls:
            tool_name = tool_c.function.name
            tool_args = tool_c.function.arguments

            print(f"Tool selected {tool_name} with args {tool_args}")

            tool_to_use = tools_dict.get(tool_name)
            if tool_to_use is None:
                raise ValueError(f"Unknown tool '{tool_name}' called by LLM")

            observation = tool_to_use(**tool_args)
            messages.append({
                "role": "tool",
                "content": str(observation)
            })

            print(f"Tool Result - {observation}")

if __name__ == "__main__":
    main("What is the price of a laptop after applying a gold discount?")

