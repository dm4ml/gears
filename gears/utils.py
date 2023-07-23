import re
import json


def extract_json(input_string: str):
    opening_bracket_match = re.search(r"\{", input_string)
    closing_bracket_match = re.search(
        r"\}", input_string[::-1]
    )  # Reverse the string to find the last occurrence
    if opening_bracket_match and closing_bracket_match:
        opening_bracket_index = opening_bracket_match.start()
        closing_bracket_index = (
            len(input_string) - closing_bracket_match.start() - 1
        )
        json_string = input_string[
            opening_bracket_index : closing_bracket_index + 1
        ]

        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError as e:
            print("Error parsing JSON:", e)
            return None
    else:
        return None
