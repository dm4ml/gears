# Primitives

This section describes the building blocks of Gears, including `Context`, `History`, `LLM`, and `Gear`.

## Context

A context is a [Pydantic model](https://docs.pydantic.dev/latest/) that represents any structured information flowing through your `Gear`s. Context attributes can be accessed in prompt prompts, and context objects are passed throughout `Gear` methods.

We use Pydantic because Pydantic automatically validates data types. To create a Pydantic object, simply subclass `pydantic.BaseModel`:

```python
from pydantic import BaseModel

class SomeContext(BaseModel, extra="allow"): # extra="allow" allows extra attributes
    some_attribute: str # This attribute is required
    some_other_attribute: str = None # This attribute is optional, and defaults to None
    error: bool = False # This attribute is optional, and defaults to False
```

You can also set default values for attributes and allow extra attributes, as shown in the above example. Creating a Pydantic object is as simple as:

```python
context = SomeContext(some_attribute="some value")
```

For more information on Pydantic and extra validators, check out the [Pydantic docs](https://docs.pydantic.dev/latest/).

## History

The `History` class is quite straightforward---it keeps track of a list of `Message` objects, where each `Message` object has `role`, `content`, and other attributes. `History` objects are initialized with an optional `system_message`, as the [`openai` docs](https://platform.openai.com/docs/guides/gpt/chat-completions-api) allow for.

To create a `History` object:

```python
from gears import History

history = History(system_message="You are a helpful assistant.")
```

You should not need to add messages to a `History` object manually, as `gears` orchestrates LLMs, histories, and your control flow logic for your.

You can print out a `History` object with:

```python
print(history)
```

## LLM

An LLM is a class that wraps an LLM API call with exponential backoff and adds request and response data to a `History` object. We support `openai` chat models out of the box, but you can easily add your own LLMs by subclassing `BaseLLM` and following [these steps](examples/customllm.md). To initialize an LLM object, you should pass in the name of the chat model you want to use, as well as any other parameters you want the LLM to use (e.g., temperature):

```python
from gears import OpenAIChat

llm = OpenAIChat("gpt-3.5-turbo", temperature=1.0)
```

You will not be calling any methods on the LLM object directly---instead, you will pass the LLM object to a `Gear` constructor, which will handle the LLM API call.

## Gear

Your control flow will live within a `Gear` object. To create a gear, you must subclass `Gear` and implement the following methods:

- `prompt`: A method that returns a jinja-formatted prompt prompt, using attributes from a context object.
- `transform`: A method that transforms the response from the LLM and initial context into a new context object.
- `switch`: A method that returns the next `Gear` to run, or `None` if the workflow should end.

Here's an example of a recursive `Gear` that asks a user to write a story, and then asks them to write another story if the first story is too long:

```python
from gears import Gear

class ExampleGear(Gear):
    def prompt(self, context: SomeContext):
        if context.error:
            return "That story was too long! Keep your story under 100 characters."
        else:
            return "Write a story about {{ context.some_attribute }}."

    def transform(self, response: dict, context: SomeContext):
        reply = response["choices"][0]["message"]["content"].strip()

        # Suppose we only want short stories, < 100 characters
        if len(reply) > 100:
            return SomeContext(some_attribute=context.some_attribute, error=True)
        else:
            return SomeContext(some_attribute=context.some_attribute, some_other_attribute=reply, error=False)

    def switch(self, context: SomeContext): # (1)!
        if context.error:
            return ExampleGear(model=llm) # (2)!
        else:
            return None # (3)!
```

1. The `switch` method is optional. If `switch` is not implemented or returns `None`, the workflow will end.
2. We want to reprompt for a story if the story is too long
3. If the story is not too long, we want to end the workflow

The way to use a `Gear` is to initialize it with an LLM object, and then call the `run` method with an initial context object and history. The `run` lifecycle is as follows:

1. `run` calls `prompt` to get a prompt prompt, passing in the context as a Jinja variable
2. `run` calls the LLM API with the prompt prompt and the history
3. `run` calls `transform` with the response from the LLM and the context object. `transform` returns a new context object. If `transform` doesn't return a Pydantic object, `gears` will throw an error.
4. `run` calls `switch` with the new context object. If `switch` returns `None`, the workflow will end. Otherwise, the returned `Gear` will be run with the new context and history.

Here's a full example of running a `Gear`:

```python
async def main():
    context = SomeContext(some_attribute="a clown")
    history = History(system_message="You are a helpful assistant.")
    llm = OpenAIChat("gpt-3.5-turbo", temperature=1.0)

    gear = ExampleGear(model=llm) # error is False by default
    final_context = await gear.run(context=context, history=history) # (1)!
    story = final_context.some_other_attribute

    print(story)
```

1. Runs until `switch` returns `None`
