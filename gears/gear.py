import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from gears.example import Example
from gears.history import History
from gears.llms.base import BaseLLM
from gears.llms.oai import OpenAIChat

logger = logging.getLogger(__name__)


class Gear(ABC):
    def __init__(
        self,
        model: BaseLLM = OpenAIChat("gpt-3.5-turbo"),
        num_retries_on_transform_error: int = 0,
    ):
        """Initializes a Gear with a LLM model.

        Args:
            model (BaseLLM): The LLM model to use. Must be an instance of a class that inherits from BaseLLM.
            num_retries_on_transform_error (int, optional): How many times to retry the LLM if the transform method raises an error. Defaults to 0.
        """
        self.model = model
        self.num_retries_on_transform_error = num_retries_on_transform_error

        # Initialize the version as the version attribute on the class
        # Used for iPython notebook extension
        self._version = getattr(self.__class__, "_version", 1)

    @property
    def version(self) -> int:
        return self._version

    def editHistory(self, context: Example, history: History) -> History:
        """Optional method to implement that edits the chat history
        after the gear is run. Default implementation does not
        modify the history.

        Args:
            context (Example): Context of the gear.
            history (History): Chat history so far.

        Returns:
            History: Edited chat history to be passed into the run method.
        """
        return history

    async def _askModel(
        self,
        prompt: str,
        edited_history: History,
        **kwargs,
    ):
        """Internal method to ask the model."""
        if prompt is None:
            # Don't run the gear
            response = None
        else:
            # Call the model
            logger.debug(f"Running model with prompt: {prompt}")
            response = await self.model.run(prompt, edited_history, **kwargs)

        return response

    async def _runWithoutSwitch(
        self,
        context: Example,
        history: History,
        model_response: Dict = None,
        **kwargs,
    ) -> Tuple[Dict, Example, History]:
        """Runs the gear with the given context and history, without executing
        a switch method.

        Args:
            context (Example): Input context for the gear. Must be a pydantic model. This is passed to the `prompt` method, which will construct the prompt for the LLM.
            history (History): A chat history.

        Returns:
            Dict: Raw response from the LLM.
            Example: The output context of the gear (result of `transform` method).
            History: The chat history after running the gear. This is a new object and does not modify the original history object.
        """
        # Transform the data from the response
        self.num_transform_tries = 0

        # Construct the prompt with the pydantic model
        context_copy = context.model_copy()
        prompt_str = self.prompt(context_copy)

        while self.num_transform_tries <= self.num_retries_on_transform_error:
            # Copy the edited history so that we don't modify the original
            history_copy = history.copy()

            model_response = (
                await self._askModel(prompt_str, history_copy, **kwargs)
                if model_response is None
                else model_response
            )

            try:
                transformed_context = self.transform(model_response, context_copy)
                break
            except NotImplementedError:
                transformed_context = context_copy
                logger.debug(
                    f"Transform method not implemented. No modifications to the context."
                )
                break
            except Exception as e:
                self.num_transform_tries += 1
                # Reset model_response
                model_response = None

                if self.num_transform_tries > self.num_retries_on_transform_error:
                    raise e
                else:
                    logger.warning(
                        f"Transform method raised an error. Asking LLM again..."
                    )

        # Verify that the structured data is a pydantic model
        if not isinstance(transformed_context, Example):
            raise TypeError("Transform must return an Example instance")

        return model_response, transformed_context, history_copy

    async def run(
        self,
        context: Example,
        history: History,
        **kwargs,
    ) -> Example:
        """Runs the gear with the given context and history. This is automatically called if executing a gear within a `switch` method; otherwise, it must be called manually for a top-level gear.

        Args:
            context (Example): Input context for the gear. Must be a pydantic model. This is passed to the `prompt` method, which will construct the prompt for the LLM.
            history (History): Chat history. This is passed to the LLM's `run` method.

        Raises:
            TypeError: If the `prompt` method does not return a string.
            TypeError: If the `transform` method does not return a pydantic model.
            TypeError: If the `switch` method does not return a Gear instance or None.

        Returns:
            Example: The output context of the gear (result of `transform` method) if no gears are chained in the `switch` method; otherwise, the output context of the last gear in the chain.
        """

        (
            model_response,
            transformed_context,
            history_copy,
        ) = await self._runWithoutSwitch(context, history, **kwargs)

        # Edit the history
        edited_history = self.editHistory(
            context=transformed_context, history=history_copy
        )

        # Load which other gear to run, if any
        try:
            child = self.switch(transformed_context)
            if isinstance(child, Gear):
                response = await child.run(
                    transformed_context, edited_history, **kwargs
                )
            elif not child:
                response = transformed_context
                logger.debug("No child gear to run. Returning response.")
            else:
                raise TypeError(f"Switch must return a Gear instance or None.")

        except NotImplementedError:
            response = transformed_context
            pass

        # Change the original history object
        history.resetFrom(edited_history)

        # If there is no child, return the transformed_context
        return response

    @abstractmethod
    def prompt(self, context: Example) -> str:
        """Function that returns a prompt, given the context.

        Args:
            context (Example): Object that will be used as context in the prompt.

        Raises:
            NotImplementedError: If the gear does not implement this method.

        Returns:
            str: Prompt that will be passed to the model.
        """
        raise NotImplementedError("Gear must implement prompt method")

    @abstractmethod
    def transform(self, response: dict, context: Example) -> Example:
        """Transforms the response from the LLM into an Example.

        Args:
            response (dict): Raw response from the LLM.
            context (Example): Input context for the gear. Must be a Example instance.

        Raises:
            NotImplementedError: If the gear does not implement this method.

        Returns:
            Example: New context for the gear. Must be a Example instance.
        """
        raise NotImplementedError("Gear must implement transform method")

    def switch(self, return_context: Example) -> Optional["Gear"]:
        """Method that determines which gear to run next. This method is called after the `transform` method with the returned context. If this method is not implemented, then the gear will return the response from the `transform` method.

        Args:
            return_context (Example): Output context from the `transform` method.

        Returns:
            Optional[Gear]: The next gear to run (an instance). If None, then the gear will return the response from the `transform` method.
        """
        raise NotImplementedError
