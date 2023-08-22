from typing import List, Union

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
    content: str = Field(..., description="The content of the message. A string.")


class History:
    """
    This class is a wrapper around a list of messages. It is used to keep track of the conversation history with an LLM.

    A history is a list of messages. Each message has a role and content. The role is something like "user" or "system". The content is a string.
    """

    def __init__(
        self,
        system_message: str = None,
        messages: List[Message] = None,
        cost: float = 0.0,
    ):
        """Constructor for the History class.

        Args:
            system_message (str, optional): System message for the LLM to use, if any. E.g., "You are a helpful assistant." Defaults to None.
            messages (List[Message], optional): List of messages to initialize the history with. Defaults to None.
            cost: (float, optional): Cost of the history. Defaults to 0.0 but should be the cost of the messages passed into the constructor.
        """
        self._value: List[Message] = messages or []
        self._cost = cost
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

    def __getitem__(self, index: Union[int, slice]):
        """Returns the message at the given index."""
        if isinstance(index, slice):
            # When sliced, return a new History instance with sliced messages
            return History(messages=self._value[index], cost=self._cost)

        return self._value[index]

    def __len__(self):
        """Returns the number of messages in the history."""
        return len(self._value)

    def add(self, message: Message):
        """Adds a message to the history. Used in LLM run methods."""

        self._value.append(message)

    def __add__(self, other: "History") -> "History":
        if not isinstance(other, History):
            raise TypeError("Can only add History to History")

        # Concatenate the messages
        combined_messages = self._value + other._value
        cost = max(self._cost, other._cost)

        return History(messages=combined_messages, cost=cost)

    def __iter__(self) -> iter:
        """An iterator over the messages in the history.

        Usage:
        ```python
        h = History()

        for message in h:
            print(message)
        ```

        Returns:
            iter: An iterator over the messages in the history.
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
        return messages_str

    def copy(self):
        return History(messages=self._value.copy(), cost=self._cost)

    def resetFrom(self, history: "History") -> None:
        self._value = history._value
        self._cost = history._cost
