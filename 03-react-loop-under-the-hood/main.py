import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

load_dotenv()

MAX_ITERATIONS = 10
MODEL = os.getenv("OLLAMA_MODEL")

@tool
def get_product_price(product: str) -> float:
    """Look up the price of product in the catalog."""
    print(f"Executing get_product_price({product})")
    prices = {"laptop": 1300, "headphones": 150, "keyboard": 90}

    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount to the price based on the customer's tier.
    Available tiers: bronze, silver, gold
    """
    print(f"Executing apply_discount({price}, {discount_tier})")
    percentages = {"bronze": 5, "silver": 12, "gold": 23}

    return price * (1 - percentages.get(discount_tier, 0) / 100)

@traceable(name="Langchain Agent Loop")
def main(prompt: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {prompt}")
    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant\n"
                "You have access to a product catalog tool"
                "and a discoint tool\n\n"
                "STRICT RULES - yoh must follow these exactly: \n"
                "1. NEVER guess or assume any product price"
                "2. Only call apply_discount AFTER you've received \n"
                "a price from get_product_price. pass the exact price \n"
                "returned by get_product_price - do OT pass a made-up number.\n"
            )
        ),
        HumanMessage(content=prompt)
    ]

    for interation in range(1, MAX_ITERATIONS + 1):
        print(f"--- ITERATION {interation} ---")
        ai_message = llm_with_tools.invoke(messages, config={"run_name": "03-react-under-the-hood"})

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"Final answer: {ai_message.content}")
            return ai_message.content

        messages.append(ai_message)

        for tool_c in tool_calls:
            tool_name = tool_c.get("name")
            tool_args = tool_c.get("args", {})
            tool_id = tool_c.get("id")

            print(f"Tool selected {tool_name} with args {tool_args}")

            tool_to_use = tools_dict.get(tool_name)
            if tool_to_use is None:
                raise ValueError(f"Unknown tool '{tool_name}' called by LLM")

            observation = tool_to_use.invoke(tool_args)
            messages.append(
                ToolMessage(content=str(observation), tool_call_id=tool_id)
            )

            print(f"Tool Result - {observation}")

if __name__ == "__main__":
    main("What is the price of a laptop after applying a gold discount?")

