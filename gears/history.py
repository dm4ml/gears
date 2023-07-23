from typing import List
from pydantic import BaseModel


class Message(BaseModel):
    class Config:
        # Set `allow_extra` to True to allow extra fields
        allow_extra = True

    role: str
    content: str


class History:
    def __init__(self, system_message: str = None):
        self._value: List[Message] = []
        if system_message:
            self.add(Message(role="system", content=system_message))

    def add(self, message: Message):
        self._value.append(message)

    def __iter__(self):
        return iter(self._value)

    def __str__(self):
        messages_str = "\n".join(
            [
                f"[{message.role.capitalize()}]: {message.content}"
                for message in self._value
            ]
        )
        return f"History:\n{messages_str}"
