from typing import Any, Optional
from pydantic import BaseModel

from jinja2 import Template
from abc import ABC, abstractmethod

from gears.history import History
from gears.llms.base import BaseLLM

import logging

logger = logging.getLogger(__name__)


class Gear(ABC):
    def __init__(
        self,
        model: BaseLLM,
    ):
        """Initializes a Gear with a LLM model.

        Args:
            model (BaseLLM): The LLM model to use. Must be an instance of a class that inherits from BaseLLM.
        """
        self.model = model

    async def run(
        self,
        context: BaseModel,
        history: History,
        **kwargs,
    ) -> BaseModel:
        """Runs the gear with the given context and history. This is automatically called if executing a gear within a `switch` method; otherwise, it must be called manually for a top-level gear.

        Args:
            context (BaseModel): Input context for the gear. Must be a pydantic model. This is passed to the `template` method, which will construct the prompt for the LLM.
            history (History): Chat history. This is passed to the LLM's `run` method.

        Raises:
            TypeError: If the `template` method does not return a string.
            TypeError: If the `transform` method does not return a pydantic model.
            TypeError: If the `switch` method does not return a Gear instance or None.

        Returns:
            BaseModel: The output context of the gear (result of `transform` method) if no gears are chained in the `switch` method; otherwise, the output context of the last gear in the chain.
        """
        # Construct the template with the pydantic model
        prompt = Template(self.template(context)).render(context=context)
        if not isinstance(prompt, str):
            raise TypeError("Template must return a string")

        # Call the model
        logger.info(f"Running model with prompt: {prompt}")
        response = await self.model.run(prompt, history, **kwargs)

        # Transform the data from the response
        response = self.transform(response, context)
        # Verify that the structured data is a pydantic model
        if not isinstance(response, BaseModel):
            raise TypeError("Transform must return a pydantic model instance")

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

    @abstractmethod
    def template(self, context: BaseModel) -> str:
        """Template for the LLM prompt. This is a jinja2 template that will be rendered with the given context.

        Args:
            context (BaseModel): Pydantic model instance that will be used to render the template.

        Raises:
            NotImplementedError: If the gear does not implement this method.

        Returns:
            str: Prompt template that will be rendered with the given context.
        """
        raise NotImplementedError("Gear must implement prompt template")

    @abstractmethod
    def transform(self, response: dict, context: BaseModel) -> BaseModel:
        """Transforms the response from the LLM into a pydantic model instance.

        Args:
            response (dict): Raw response from the LLM.
            context (BaseModel): Input context for the gear. Must be a pydantic model.

        Raises:
            NotImplementedError: If the gear does not implement this method.

        Returns:
            BaseModel: New context for the gear. Must be a pydantic model instance.
        """
        raise NotImplementedError("Gear must implement transform method")

    def switch(self, return_context: BaseModel) -> Optional["Gear"]:
        """Method that determines which gear to run next. This method is called after the `transform` method with the returned context. If this method is not implemented, then the gear will return the response from the `transform` method.

        Args:
            return_context (BaseModel): Output context from the `transform` method.

        Returns:
            Optional[Gear]: The next gear to run (an instance). If None, then the gear will return the response from the `transform` method.
        """
        raise NotImplementedError
