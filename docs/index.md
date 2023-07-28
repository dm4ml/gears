# Welcome to Gears

Gears is a lightweight framework for writing control flow with LLMs with **full control over your prompts**. It allows you to build complex chains of actions and conditions, and execute them in a single call.

## Why Gears?

Gears is so minimal---it is simply a wrapper around an LLM API call that:

- Allows you to specify your prompts as Jinja templates
- Automatically handles LLM API failures with exponential backoff
- Allows you to specify control flow, based on LLM responses, in a simple, declarative way

We will never suffer the bloat of a venture-backed open source project, and we are committed to _not_ growing the codebase beyond what is necessary to support the above features.

## Installation

Gears is available on PyPI, and can be installed with pip:

```bash
pip install gearsllm
```
