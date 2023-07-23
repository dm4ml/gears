from typing import Any
from pydantic import BaseModel

from jinja2 import Template
import inspect

from gears.history import History
from gears.llms.base import BaseLLM

import logging

logger = logging.getLogger(__name__)


class Gear:
    def __init__(
        self,
        model: BaseLLM,
    ):
        self.model = model
        self.template = Template(self.template())

    async def run(self, data: BaseModel, history: History, **kwargs):
        # Construct the template with the pydantic model
        prompt = self.template.render(data.dict())

        # Call the model
        logger.info(f"Running model with prompt: {prompt}")
        reply = await self.model.run(prompt, history, **kwargs)

        # Transform the data from the reply
        try:
            relevant_args = inspect.getfullargspec(self.transform)[0]
            relevant_data_dict = {
                key: value
                for key, value in data.dict().items()
                if key in relevant_args
            }
            reply = self.transform(reply, **relevant_data_dict)
            # Verify that the structured data is a pydantic model
            if not isinstance(reply, BaseModel):
                raise TypeError(
                    "Transform must return a pydantic model instance"
                )
        except NotImplementedError:
            pass

        # Load which other gear to run, if any
        try:
            child = self.switch(reply)
            return await child.run(reply, history, **kwargs)
        except NotImplementedError:
            pass

        # If there is no child, return the reply
        return reply

    def template(self) -> str:
        raise NotImplementedError(
            "Gear must implement prompt template if relying on a model"
        )

    def transform(self, reply: str, **kwargs) -> BaseModel:
        raise NotImplementedError

    def switch(self, reply: BaseModel, **kwargs) -> "Gear":
        raise NotImplementedError
