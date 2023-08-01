# Welcome to Gears

Gears is a lightweight framework for writing control flow with LLMs with **full control over your prompts**. It allows you to build complex chains of actions and conditions, and execute them in a single call.

## Why Gears?

Gears is so minimal; it is simply a wrapper around an LLM API call that:

- Allows you to specify your prompts as [Jinja templates](https://jinja.palletsprojects.com/en/3.1.x/) and inputs as [Pydantic models](https://docs.pydantic.dev/latest/)
- Automatically handles LLM API failures with [exponential backoff](https://tenacity.readthedocs.io/en/latest/)
- Allows you to specify control flow, based on LLM responses, in a simple, declarative way

But the real selling point is that _we will not suffer the bloat of a venture-backed open source project; we are committed to **not** growing the codebase beyond what is necessary to support the above features._

## Installation

Gears is available on PyPI, and can be installed with pip:

```bash
pip install gearsllm
```

## ToDos

- [ ] Run examples in docs + add outputs
- [ ] Add pre-commit hooks with black & isort
