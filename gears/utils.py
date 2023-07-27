import json

from typing import Generator, List, Dict, Union


def extract_json(
    text, decoder=json.JSONDecoder()
) -> Generator[Union[Dict, List], None, None]:
    """
    Generates all JSON objects or arrays in a string representing the output of an LLM. Copied from ChatGPT.

    Args:
        text (str): The string to parse.

    Returns:
        Generator[Union[Dict, List], None, None]: A generator of JSON objects or arrays.
    """
    pos = 0
    while True:
        match = text.find("{", pos)
        match2 = text.find("[", pos)
        # Find the earliest opening bracket
        if (match != -1 and match < match2) or match2 == -1:
            start_pos = match
        else:
            start_pos = match2

        if start_pos == -1:
            break
        try:
            result, index = decoder.raw_decode(text[start_pos:])
            yield result
            pos = start_pos + index
        except json.JSONDecodeError:
            pos = start_pos + 1


def extract_first_json(text: str) -> Union[Dict, List]:
    """
    Extracts the first JSON object or array in a string representing the output of an LLM.

    Args:
        text (str): The string to parse.

    Returns:
        Union[Dict, List]: The first JSON object or array.
    """
    try:
        return next(extract_json(text))
    except StopIteration:
        raise StopIteration(f"No JSON found in text: {text}")
