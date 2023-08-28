"""
This file contains code for a Session object. A Session keeps track
of examples, multiple gears, and intermediate results when prototyping
chains of gears.
"""
import asyncio
import hashlib
import logging
from collections import namedtuple
from typing import Dict, List

from rich.logging import RichHandler

from gears.example import Example
from gears.gear import Gear
from gears.history import History

Intermediate = namedtuple(
    "Intermediate",
    [
        "current_context",
        "result_context",
        "current_history",
        "new_history",
    ],
)


def hash_example(obj: Example) -> str:
    # Convert the Pydantic object to a standardized JSON string
    serialized_data = obj.model_dump_json()

    # Compute the hash
    hash_object = hashlib.md5(serialized_data.encode())
    return hash_object.hexdigest()


class Session:
    def __init__(self, logging_level: str = logging.INFO) -> None:
        """Creates a Session object."""
        self.__examples = []
        self.__root_gear = None
        self.__intermediates = {}
        self.__cost = 0.0

        # Set logging level
        logging.basicConfig(
            level=logging_level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler()],
        )
        self._logger = logging.getLogger("rich")

    @property
    def cost(self) -> float:
        return self.__cost

    def add_example(self, example: Example) -> None:
        """Adds an example to the session.

        Args:
            example (Example): Example to add to the session.
        """
        self.__examples.append(example)

    def add_examples(self, examples: List[Example]) -> None:
        """Adds a list of examples to the session.

        Args:
            examples (list[Example]): List of examples to add to the session.
        """
        self.__examples.extend(examples)

    @property
    def examples(self) -> List[Example]:
        """Returns the examples in the session.

        Returns:
            list[Example]: List of examples in the session.
        """
        return self.__examples

    def register_root(self, root_gear: Gear, bypass_error: bool = False) -> None:
        """Registers the root gear in the session.
        Throws an error if there is already a root gear registered
        and bypass_error is False.

        Args:
            root_gear (Gear): Root gear in the session.
            bypass_error (bool, optional): Whether to bypass the error if there is already a root gear registered. Defaults to False.
        """
        if self.__root_gear and not bypass_error:
            raise ValueError("Root gear already registered.")

        self.__root_gear = root_gear

    def unregister_root(self) -> None:
        """Unregisters the root gear in the session.
        Throws an error if there is no root gear registered.
        """
        if not self.__root_gear:
            raise ValueError("No root gear registered.")

        self.__root_gear = None

    @property
    def root(self) -> Gear:
        """Returns the root gear in the session.

        Returns:
            Gear: Root gear in the session.
        """
        return self.__root_gear

    async def _run_example(self, example: Example) -> Example:
        """Internal method to process a single example asynchronously."""
        current_gear = self.root
        current_context = example.model_copy()
        current_history = History()

        # Recursively run the gears
        while current_gear:
            # Hash the current context into a key prefix
            context_hash = hash_example(current_context)

            key = (
                context_hash,
                current_gear.__class__.__name__,
                current_gear.version,
            )
            if key not in self.__intermediates:
                self._logger.info(
                    f"Running {current_gear.__class__.__name__} on example {current_context.id}."
                )
                (result_context, new_history,) = await current_gear._runWithoutSwitch(
                    current_context, current_history
                )

                self.__intermediates[key] = Intermediate(
                    current_context=current_context,
                    result_context=result_context,
                    current_history=current_history,
                    new_history=new_history,
                )

                # Increment cost
                self.__cost += new_history.cost - current_history.cost
            else:
                self._logger.debug(
                    f"Found intermediate result for {current_gear.__class__.__name__} on example {current_context.id}. Not running again."
                )

            # Load which other gear to run, if any
            # We should always run this to detect version changes
            try:
                child = current_gear.switch(self.__intermediates[key].result_context)
                if child and not isinstance(child, Gear):
                    raise TypeError(f"Switch must return a Gear instance or None.")

            except NotImplementedError:
                child = None

            # Get the next gear
            current_gear = child
            current_context = self.__intermediates[key].result_context
            current_history = self.__intermediates[key].new_history

        # Return the final result for this example
        return current_context

    async def run(self) -> List[Example]:
        """
        Runs the root gear on the examples in the session and stores the
        results in the intermediates dictionary.
        """
        start_cost = self.cost
        # Process examples in parallel using asyncio.gather
        results = await asyncio.gather(*[self._run_example(ex) for ex in self.examples])

        # If cost changed, log it
        if self.cost != start_cost:
            self._logger.info(
                f"This run cost ${self.cost - start_cost:.4f}. The total cost of the session so far is ${self.cost:.4f}."
            )
        else:
            self._logger.info(
                f"No new gears ran. The total cost of the session so far is ${self.cost:.4f}."
            )

        return results

    def is_stale(self) -> bool:
        """
        Checks if the session is stale (i.e., examples don't have the latest versions of gears run on them).
        """
        for example in self.examples:
            current_gear = self.root
            current_context = example.model_copy()

            # Recursively run the gears
            while current_gear:
                # Hash the current context into a key prefix
                context_hash = hash_example(current_context)

                key = (
                    context_hash,
                    current_gear.__class__.__name__,
                    current_gear.version,
                )
                if key not in self.__intermediates:
                    return True

                # Load which other gear to run, if any
                # We should always run this to detect version changes
                try:
                    child = current_gear.switch(
                        self.__intermediates[key].result_context
                    )
                    if child and not isinstance(child, Gear):
                        raise TypeError(f"Switch must return a Gear instance or None.")

                except NotImplementedError:
                    child = None

                # Get the next gear
                current_gear = child
                current_context = self.__intermediates[key].result_context

        return False

    @property
    def intermediates(self) -> Dict[str, Intermediate]:
        return self.__intermediates.copy()
