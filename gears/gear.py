import logging
from abc import ABC, abstractmethod
from typing import Optional

from jinja2 import Template
from pydantic import BaseModel

from gears.history import History
from gears.llms.base import BaseLLM

logger = logging.getLogger(__name__)


class Gear(ABC):
    def __init__(
        self,
        model: BaseLLM,
        num_retries_on_transform_error: int = 0,
    ):
        """Initializes a Gear with a LLM model.

        Args:
            model (BaseLLM): The LLM model to use. Must be an instance of a class that inherits from BaseLLM.
            num_retries_on_transform_error (int, optional): How many times to retry the LLM if the transform method raises an error. Defaults to 0.
        """
        self.model = model
        self.num_retries_on_transform_error = num_retries_on_transform_error

    def editHistory(self, context: BaseModel, history: History) -> History:
        """Optional method to implement that edits the chat history
        before the gear is run. Default implementation does not
        modify the history.

        Args:
            context (BaseModel): Context of the gear.
            history (History): Chat history so far.

        Returns:
            History: Edited chat history to be passed into the run method.
        """
        return history

    async def _askModel(
        self,
        template_str: str,
        edited_history: History,
        context: BaseModel,
        **kwargs,
    ):
        """Internal method to ask the model."""
        if template_str is None:
            # Don't run the gear
            response = None
        else:
            prompt = Template(template_str).render(context=context)
            if not isinstance(prompt, str):
                raise TypeError("Template must return a string")

            # Call the model
            logger.debug(f"Running model with prompt: {prompt}")
            response = await self.model.run(prompt, edited_history, **kwargs)

        return response

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

        # Transform the data from the response
        num_transform_tries = 0

        # Construct the template with the pydantic model
        template_str = self.template(context)
        edited_history = self.editHistory(context=context, history=history)

        while num_transform_tries <= self.num_retries_on_transform_error:
            # Copy the edited history so that we don't modify the original
            edited_history_copy = edited_history.copy()

            response = await self._askModel(
                template_str, edited_history_copy, context, **kwargs
            )
            try:
                response = self.transform(response, context)
                break
            except Exception as e:
                num_transform_tries += 1

                if num_transform_tries > self.num_retries_on_transform_error:
                    raise e
                else:
                    logger.warning(
                        f"Transform method raised an error. Asking LLM again..."
                    )

        # Verify that the structured data is a pydantic model
        if not isinstance(response, BaseModel):
            raise TypeError("Transform must return a pydantic model instance")

        # Load which other gear to run, if any
        try:
            child = self.switch(response)
            if isinstance(child, Gear):
                response = await child.run(response, edited_history_copy, **kwargs)
            elif not child:
                logger.debug("No child gear to run. Returning response.")
            else:
                raise TypeError(f"Switch must return a Gear instance or None.")

        except NotImplementedError:
            pass

        # Change the original history object
        history.resetFrom(edited_history_copy)

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
