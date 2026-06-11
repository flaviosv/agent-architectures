import os
import re
from pathlib import Path
import ollama
import inspect

from dotenv import load_dotenv
from langsmith import traceable

load_dotenv(Path(__file__).parent / "../.env")

MAX_ITERATIONS = 10
MODEL = os.getenv("OLLAMA_MODEL")

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of product in the catalog."""
    print(f"Executing get_product_price({product})")
    prices = {"laptop": 1300, "headphones": 150, "keyboard": 90}

    return prices.get(product, 0)

@traceable(run_type="tool")
def apply_discount(price, discount_tier: str) -> float:
    """
    Apply a discount to the price based on the customer's tier.
    Available tiers: bronze, silver, gold
    """
    price = float(price)
    print(f"Executing apply_discount({price}, {discount_tier})")
    percentages = {"bronze": 5, "silver": 12, "gold": 23}

    return price * (1 - percentages.get(discount_tier, 0) / 100)


traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}
def get_tool_description(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function)
        descriptions.append(f"{tool_name}{signature} - {docstring}")

    return "\n".join(descriptions)

tool_description = get_tool_description(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""
You are a helpful shopping assistant
You have access to a product catalog tool"
and a discount tool

STRICT RULES - yoh must follow these exactly:
1. NEVER guess or assume any product price"
2. Only call apply_discount AFTER you've received
a price from get_product_price. pass the exact price
returned by get_product_price - do OT pass a made-up number.

Answer the following questions as best you can. You have access to the following tools:

{tool_description}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{prompt}}
Thought:
"""

@traceable(name="Langchain Agent Loop")
def main(prompt: str):
    print(f"Question: {prompt}")
    prompt = react_prompt.format(prompt=prompt)
    scratchpad = ""

    for interation in range(1, MAX_ITERATIONS + 1):
        print(f"--- ITERATION {interation} ---")
        full_prompt = prompt + "\n" + scratchpad

        response = ollama_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation"], "temperature": 0}
        )
        output = response.message.content

        print(f"LLM Output: {output}")

        final_answer_match = re.search(r"Final answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"Final answer: {final_answer}")

            return final_answer

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print("Error finding tool")
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        if not tool_name in tool_names:
            observation = f"Unknown tool '{tool_name}' called by LLM, Available tools: {', '.join(tool_names)}"
        else:
            observation = str(tools[tool_name](*args))

        scratchpad += f"{output}\nObservation: {observation}\nThought:"

if __name__ == "__main__":
    main("What is the price of a laptop after applying a gold discount?")

