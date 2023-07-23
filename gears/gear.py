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
        prompt = self.template.render(data.model_dump())

        # Call the model
        logger.info(f"Running model with prompt: {prompt}")
        response = await self.model.run(prompt, history, **kwargs)

        # Transform the data from the response
        try:
            relevant_args = inspect.getfullargspec(self.transform)[0]
            relevant_data_dict = {
                key: value
                for key, value in data.model_dump().items()
                if key in relevant_args
            }
            response = self.transform(response, **relevant_data_dict)
            # Verify that the structured data is a pydantic model
            if not isinstance(response, BaseModel):
                raise TypeError(
                    "Transform must return a pydantic model instance"
                )
        except NotImplementedError:
            pass

        # Load which other gear to run, if any
        try:
            child = self.switch(response)
            return await child.run(response, history, **kwargs)
        except NotImplementedError:
            pass

        # If there is no child, return the response
        return response

    def template(self) -> str:
        raise NotImplementedError("Gear must implement prompt template")

    def transform(self, response: dict, **kwargs) -> BaseModel:
        raise NotImplementedError

    def switch(self, response: dict, **kwargs) -> "Gear":
        raise NotImplementedError
