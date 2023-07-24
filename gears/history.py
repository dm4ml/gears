from typing import List
from pydantic import BaseModel, Field


class Message(BaseModel, extra="allow"):
    """This class is a wrapper around a message. It is a building block of a history.

    Attributes:
        role (str): The role of the message. Something like "user" or "system".
        content (str): The content of the message. A string.
    """

    role: str = Field(
        ...,
        description="The role of the message. Something like 'user' or 'system'.",
    )
    content: str = Field(
        ..., description="The content of the message. A string."
    )


class History:
    """
    This class is a wrapper around a list of messages. It is used to keep track of the conversation history with an LLM.

    A history is a list of messages. Each message has a role and content. The role is something like "user" or "system". The content is a string.
    """

    def __init__(self, system_message: str = None):
        """Constructor for the History class.

        Args:
            system_message (str, optional): System message for the LLM to use, if any. E.g., "You are a helpful assistant." Defaults to None.
        """
        self._value: List[Message] = []
        self._cost = 0.0
        if system_message:
            self.add(Message(role="system", content=system_message))

    @property
    def cost(self):
        return self._cost

    def increment_cost(self, cost: float):
        """Increments the cost of the history.

        Args:
            cost (float): Float representing the cost to increment by (in dollars)
        """
        self._cost += cost

    def __getitem__(self, index: int):
        """Returns the message at the given index."""
        return self._value[index]

    def __len__(self):
        """Returns the number of messages in the history."""
        return len(self._value)

    def add(self, message: Message):
        """Adds a message to the history. Used in LLM run methods."""

        self._value.append(message)

    def __iter__(self):
        """An iterator over the messages in the history.

        Usage:
        ```python
        h = History()

        for message in h:
            print(message)
        ```

        Returns:
            _type_: _description_
        """
        return iter(self._value)

    def __str__(self):
        """String representation of the history."""
        messages_str = "\n".join(
            [
                f"[{message.role.capitalize()}]: {message.content}"
                for message in self._value
            ]
        )
        return f"History:\n{messages_str}"
