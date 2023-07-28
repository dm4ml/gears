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
        self.template = self.template()

    async def run(
        self,
        data: BaseModel,
        history: History,
        **kwargs,
    ):
        # Construct the template with the pydantic model
        try:
            items = data.model_dump()
        except AttributeError:
            items = data.dict()

        prompt = Template(self.template).render(items)

        # Call the model
        logger.info(f"Running model with prompt: {prompt}")
        response = await self.model.run(prompt, history, **kwargs)

        # Transform the data from the response
        try:
            response = self.transform(response, data)
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
            if isinstance(child, Gear):
                return await child.run(response, history, **kwargs)
            elif not child:
                logger.info("No child gear to run. Returning response.")
                return response
            else:
                raise TypeError(f"Switch must return a Gear instance or None.")

        except NotImplementedError:
            pass

        # If there is no child, return the response
        return response

    def template(self) -> str:
        raise NotImplementedError("Gear must implement prompt template")

    def transform(self, response: dict, **kwargs) -> BaseModel:
        raise NotImplementedError

    def switch(self, response: BaseModel, **kwargs) -> "Gear":
        raise NotImplementedError
