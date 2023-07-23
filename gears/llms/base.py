from abc import ABC, abstractmethod

from gears.history import History


class BaseLLM(ABC):
    @abstractmethod
    async def run(
        prompt: str,
        history: History,
        message_kwargs: dict = {},
    ):
        pass
