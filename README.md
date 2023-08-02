# Welcome to Gears

Gears is a lightweight framework for writing control flow with LLMs with **full control over your prompts**. It allows you to build complex chains of actions and conditions, and execute them in a single call.

## Why Gears?

Gears is so minimal; it is simply a wrapper around an LLM API call that:

- Allows you to specify your prompts as [Jinja templates](https://jinja.palletsprojects.com/en/3.1.x/) and inputs as [Pydantic models](https://docs.pydantic.dev/latest/)
- Automatically handles LLM API failures with [exponential backoff](https://tenacity.readthedocs.io/en/latest/)
- Allows you to specify control flow, based on LLM responses, in a simple, declarative way

But the real selling point is that _we are committed to **not** growing the codebase beyond what is necessary to support the above features._ (We are not venture-backed and do not intend to be.)

## Installation

Gears is available on PyPI, and can be installed with pip:

```bash
pip install gearsllm
```

## Dependencies

Gears has the following dependencies:

- `python>=3.9`
- `pydantic`
- `jinja2`
- `tenacity`
- `openai`

## ToDos

- [ ] Add pre-commit hooks with black & isort
