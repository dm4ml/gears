"""
This file tests the utils of JSON extraction.
Created by chat gpt
"""
from gears.utils import extract_first_json, extract_json
import pytest
import json


def test_extract_json_valid():
    text = '{"key1": "value1"}[{"key2": "value2"}]{"key3": "value3"}'
    results = list(extract_json(text))
    assert results == [
        {"key1": "value1"},
        [{"key2": "value2"}],
        {"key3": "value3"},
    ]


def test_extract_json_partial_invalid():
    text = '{"key1": "value1"}invalid[{"key2": "value2"}]{"key3": "value3"}'
    results = list(extract_json(text))
    assert results == [
        {"key1": "value1"},
        [{"key2": "value2"}],
        {"key3": "value3"},
    ]


def test_extract_first_json_valid():
    text = '{"key1": "value1"}[{"key2": "value2"}]{"key3": "value3"}'
    result = extract_first_json(text)
    assert result == {"key1": "value1"}


def test_extract_first_json_partial_invalid():
    text = '{"key1": "value1"}invalid[{"key2": "value2"}]{"key3": "value3"}'
    result = extract_first_json(text)
    assert result == {"key1": "value1"}


def test_extract_first_json_empty():
    text = ""
    with pytest.raises(StopIteration):
        extract_first_json(text)
