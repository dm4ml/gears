"""
This file contains the OpenAI chat API wrappers.
"""
from gears.history import History, Message
from gears.llms.base import BaseLLM
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import openai

import logging

logger = logging.getLogger(__name__)


def retry_after_attempts(max_retries: int):
    return retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(max_retries),
    )


class OpenAIChat(BaseLLM):
    def __init__(
        self, model: str = "gpt-3.5-turbo", max_retries: int = 3, **kwargs
    ):
        self.model = model
        self.api_kwargs = kwargs
        self.max_retries = max_retries

    def retry_policy(self):
        return retry(
            wait=wait_random_exponential(min=1, max=60),
            stop=stop_after_attempt(self.max_retries),
        )

    async def chat_api_call(self, request: dict):
        retry_decorator = self.retry_policy()
        decorated_func = retry_decorator(self._chat_api_call_impl)
        return await decorated_func(request)

    async def _chat_api_call_impl(self, request: dict):
        try:
            response = await openai.ChatCompletion.acreate(**request)
            return response
        except Exception as e:
            logger.warning(
                f"Exception when calling OpenAI occurred: {e}. Retrying..."
            )
            raise

    async def run(
        self,
        prompt: str,
        history: History,
        message_kwargs: dict = {},
    ) -> str:
        # Construct chat history
        curr_message = Message(role="user", content=prompt, **message_kwargs)
        history.add(curr_message)
        messages = [m.dict() for m in history]
        request = {
            "model": self.model,
            "messages": messages,
            **self.api_kwargs,
        }
        response = await self.chat_api_call(request)
        returned_message = response["choices"][0]["message"]["content"].strip()
        history.add(Message(role="system", content=returned_message))
        return returned_message
