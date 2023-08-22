from gears.gear import Gear
from pydantic import BaseModel
from gears import Gear, History
from gears.llms import Echo
from gears.utils import extract_first_json
import pytest
import random


class Input(BaseModel):
    name: str


class Output(BaseModel):
    name: str
    completion: str


echo_model = Echo()


@pytest.mark.asyncio
async def test_single_gear():
    class TestGear(Gear):
        def template(self, context):
            return "Hello {{ context.name }}"

        def transform(self, response: str, inp: Input):
            return Output(name=inp.name, completion=response)

    gear = TestGear(echo_model)
    history = History()
    result = await gear.run(Input(name="world"), history)
    assert result.name == "world"
    assert result.completion == "Hello world"
    assert len(history) == 2


@pytest.mark.asyncio
async def test_transform_error():
    class TestGear(Gear):
        def __init__(self, model, num_retries_on_transform_error: int = 0):
            self.counter = 0
            super().__init__(
                model,
                num_retries_on_transform_error=num_retries_on_transform_error,
            )

        def template(self, context):
            return '{"hello:}'

        def transform(self, response: str, inp: Input):
            # Extract the Json

            if self.counter == 0:
                self.counter += 1
                obj = extract_first_json(response)

            else:
                return Output(name=inp.name, completion="Hello world")

    gear = TestGear(echo_model, num_retries_on_transform_error=1)
    history = History()
    result = await gear.run(Input(name="world"), history)
    assert len(history) == 2


@pytest.mark.asyncio
async def test_two_gears_list():
    class TestGear(Gear):
        def template(self, context):
            return "Hello {{ context.name }}"

        def transform(self, response: str, inp: Input):
            return Output(name=inp.name, completion=response)

        def switch(self, response) -> Gear:
            return TestGear2(echo_model)

    class TestGear2(Gear):
        def template(self, context):
            return "Bye {{ context.name }}"

        def transform(self, response: str, out: Output):
            return Output(name=out.name, completion=response)

    gear = TestGear(echo_model)
    history = History()
    result = await gear.run(Input(name="world"), history)
    assert result.name == "world"
    assert result.completion == "Bye world"
    assert len(history) == 4


@pytest.mark.asyncio
async def test_editable_history():
    class TestGear(Gear):
        def template(self, context):
            return "Hello {{ context.name }}"

        def transform(self, response: str, inp: Input):
            return Output(name=inp.name, completion=response)

        def switch(self, response) -> Gear:
            return TestGear2(echo_model)

    class TestGear2(Gear):
        def template(self, context):
            return "Bye {{ context.name }}"

        def editHistory(self, history: History, context: BaseModel) -> History:
            # Delete the last 2 messages
            return history[:-2]

        def transform(self, response: str, out: Output):
            return Output(name=out.name, completion=response)

    gear = TestGear(echo_model)
    history = History()
    result = await gear.run(Input(name="world"), history)
    assert result.name == "world"
    assert result.completion == "Bye world"
    assert len(history) == 2


@pytest.mark.asyncio
async def test_three_gears_fork():
    class TestGear(Gear):
        def template(self, context):
            return "Hello {{ context.name }}"

        def transform(self, response: str, inp: Input):
            return Output(name=inp.name, completion=response)

        def switch(self, response) -> Gear:
            if response.name == "left":
                return TestGearLeft(echo_model)
            elif response.name == "right":
                return TestGearRight(echo_model)
            elif response.name == "stop":
                return None
            else:
                raise ValueError("Name must be left or right")

    class TestGearLeft(Gear):
        def template(self, context):
            return "Left bye {{ context.name }}"

        def transform(self, response: str, out: Output):
            return Output(name=out.name, completion=response)

    class TestGearRight(Gear):
        def template(self, context):
            return "Right bye {{ context.name }}"

        def transform(self, response: str, out: Output):
            return Output(name=out.name, completion=response)

    # Try left
    gear = TestGear(echo_model)
    history = History()
    result = await gear.run(Input(name="left"), history)
    assert result.name == "left"
    assert result.completion == "Left bye left"
    assert len(history) == 4

    # Try right
    gear = TestGear(echo_model)
    history = History()
    result = await gear.run(Input(name="right"), history)
    assert result.name == "right"
    assert result.completion == "Right bye right"
    assert len(history) == 4

    # Try stop
    gear = TestGear(echo_model)
    history = History()
    result = await gear.run(Input(name="stop"), history)
    assert result.name == "stop"
    assert result.completion == "Hello stop"
    assert len(history) == 2

    # Try wrong
    gear = TestGear(echo_model)
    history = History()
    with pytest.raises(ValueError):
        result = await gear.run(Input(name="wrong"), history)


@pytest.mark.asyncio
async def test_noop():
    class TestGear(Gear):
        def template(self, context):
            return None

        def transform(self, response: str, inp: Input):
            assert response is None
            return Output(name=inp.name, completion="")

    gear = TestGear(echo_model)
    history = History()
    result = await gear.run(Input(name="left"), history)

    assert result.name == "left"
    assert result.completion == ""
