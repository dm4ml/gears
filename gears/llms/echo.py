"""
This file shows how to create a custom "non-LLM" that can be used in the LLM pipeline. You can replace the logic with any python logic. Right now
it just returns the prompt. This is useful for testing the LLM pipeline without actually using an LLM.
"""
from gears.history import History, Message
from gears.llms.base import BaseLLM
from typing import Any, Callable

import logging

logger = logging.getLogger(__name__)


class Echo(BaseLLM):
    async def run(
        self,
        prompt: str,
        history: History,
        message_kwargs: dict = {},
    ) -> Any:
        # Construct chat history
        curr_message = Message(role="user", content=prompt, **message_kwargs)
        history.add(curr_message)

        # Run some function instead of an actual LLM

        history.add(
            Message(
                role="assistant",
                content=prompt,
            )
        )

        return prompt
