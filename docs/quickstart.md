# Quickstart

If you haven't already, download Gears from PyPI:

```bash
pip install gearsllm
```

This quickstart will walk you through setting up a workflow with Gears. A `Gear` is, simply, a class that wraps an LLM API call.

## Configure your LLM

At the moment, Gears only supports chat-based models from `openai` and Azure `openai`. If you are using plain `openai`, simply configure your API key in your environment as described in their [Python client library docs](https://platform.openai.com/docs/libraries/python-library). If you are using Azure `openai`, good luck finding useful documentation---[this](https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart?tabs=command-line&pivots=programming-language-python) is the best I've found.

## Create a Context

A context is, simply, a Pydantic model that represents any structured information flowing through your `Gear`s. For example, if you are building a `Gear` that takes a user's name and returns a greeting, you might define a context like this:

```python
from pydantic import BaseModel

class GreetingContext(BaseModel):
    name: str
    greeting: str = None # This will be set by the LLM flow
```

## Connect to LLM

We'll use `openai`'s `gpt-3.5-turbo` for this example:

```python
from gears import OpenAIChat

llm = OpenAIChat("gpt-3.5-turbo", temperature=1.0)

```

## Create a Gear

A `Gear` is a class that wraps an LLM API call. A `Gear` has a jinja-formatted prompt template, `transform` method to transform the output of the LLM into a context, and `switch` method to select the next `Gear` to run. `Gear` objects are initialized with an LLM object.

Here's a set of gears to provide a structured greeting:

```python
from gears import Gear

class ComplimentGear(Gear):
    def template(self, context: GreetingContext):
        return "Write a sentence to make someone named {{ context.name }} feel good about their character."

    def transform(self, response: dict, context: GreetingContext):
        reply = response["choices"][0]["message"]["content"].strip()

        return GreetingContext(name=context.name, greeting=reply)

    def switch(self, context: GreetingContext):
        return HumanityGear(context=context)

class HumanityGear(Gear):
    def template(self, context: GreetingContext):
        return "Now write a sentence that will make {{ context.name }} feel good about humanity."

    def transform(self, response: dict, context: GreetingContext):
        reply = response["choices"][0]["message"]["content"].strip()

        return GreetingContext(name=context.name, greeting=context.greeting + " " + reply)
```

When running a `Gear`, Gears will automatically call the `transform` method on the response from the LLM, and then call the `switch` method to determine the next `Gear` to run. If `switch` returns `None` or is not implemented, the workflow will end.

## Create a History

A `History` is an object that wraps messages that flow through gears, like a chat history. Simply initialize a `History` object as follows:

```python
from gears import History

history = History(system_message = "You are an optimistic person who likes to make people feel good about themselves.")
```

## Run your Workflow

To run a workflow, initialize a specific context, the top-level gear, and call the top-level gear's `run` method:

```python
import asyncio

async def main():
    context = GreetingContext(name="Alice")
    cgear = ComplimentGear(llm)
    result_context = await cgear.run(context, history)
    print(f"Greeting: {result_context.greeting}")
    print(f"Chat history:\n{history}")
    print(f"Cost: {history.cost}")

asyncio.run(main())
```

This will print:

```
Greeting: Alice, your kindness and empathy towards others truly sets you apart and makes the world a better place. Alice, your unwavering belief in the goodness of humanity constantly reminds us that there is still so much compassion and love in the world.
Chat history:
[System]: You are an optimistic person who likes to make people feel good about themselves.
[User]: Write a sentence to make someone named Alice feel good about their character.
[Assistant]: Alice, your kindness and empathy towards others truly sets you apart and makes the world a better place.
[User]: Now write a sentence that will make Alice feel good about humanity.
[Assistant]: Alice, your unwavering belief in the goodness of humanity constantly reminds us that there is still so much compassion and love in the world.
Cost: 0.00027749999999999997
```

That's it!
