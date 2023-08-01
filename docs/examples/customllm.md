# Creating a Custom LLM for Gears

Gears supports `openai`'s chat models (as well as the Azure version) out of the box, but you can easily add your own LLMs by subclassing `BaseLLM` and following the steps below.

To create a custom LLM, you should subclass `BaseLLM`:

```python
class BaseLLM(ABC):
    @abstractmethod
    async def run(
        prompt: str,
        history: History,
        message_kwargs: dict = {},
    ):
        pass
```

The `run` method should take in a prompt, a `History` object, and any other keyword arguments you want to pass to the LLM. The `run` method should return a response from the LLM, which will be passed to the `transform` method of the `Gear` that called the LLM. The `run` method should also update the `History` object with the request and response data, and update the history's cost.

Here is the `run` method for `OpenAIChat`, which constructs a chat history and calls the OpenAI chat API:

```python
async def run(
        self,
        prompt: str,
        history: History,
        **message_kwargs: Any,
    ) -> Any:
        # Construct chat history
        curr_message = Message(role="user", content=prompt, **message_kwargs)
        history.add(curr_message)
        try:
            messages = [m.model_dump() for m in history]
        except AttributeError:
            messages = [m.dict() for m in history]
        request = {
            "model": self.model,
            "messages": messages,
            **self.api_kwargs,
        }
        response = await self.chat_api_call(request) # (1)!
        try:
            returned_message = response["choices"][0]["message"]["content"]
            history.add(
                Message(
                    role=response["choices"][0]["message"]["role"],
                    content=returned_message,
                )
            ) # (2)!
        except KeyError:
            logger.error(
                f"Could not find message and/or role in response: {response}"
            )

        # Increment cost
        try:
            prompt_tokens = response["usage"]["prompt_tokens"]
            completion_tokens = response["usage"]["completion_tokens"]

            cost = (
                OPENAI_PRICING_MAP[self.model_base]["prompt_tokens"]
                * prompt_tokens
                + OPENAI_PRICING_MAP[self.model_base]["completion_tokens"]
                * completion_tokens
            )
            history.increment_cost(cost) # (3)!
        except KeyError:
            logger.error(
                f"Could not find pricing for model {self.model}. Not incrementing cost."
            )

        return response
```

1. `chat_api_call` is a helper method that calls the OpenAI chat API with the request constructed above.
2. `history.add` is a helper method that adds a message to the history. You must add messages to the history to be used in downstream Gears.
3. `history.increment_cost` is a helper method that increments the cost of the history. This way, after a workflow is run, you can see how much it cost to run the workflow.
