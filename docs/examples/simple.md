# Text to Executable SQL (Simple Control Flow)

While many natural language to SQL systems these days can generate compilable SQL, it's hard for them to generate _executable_ SQL. For example, the following query is not executable if the `users` table does not have a `name` column:

```
SELECT * FROM users WHERE name = "Alice"
```

One of the areas where `gears` shines is the ability to validate LLM output and dynamically decide what to do. In this tutorial, we will build a simple `Gear` that generates executable SQL from a natural language query.

## 0. Downloads

Assuming you've already configured your `openai` keys, download the Python libraries we'll be using in this tutorial:

```bash
pip install duckdb
```

We'll be using the NYC Taxicab dataset, described in [this DuckDB blog post](https://duckdb.org/2021/06/25/querying-parquet.html). Download it using `wget` like so:

```bash
wget https://github.com/cwida/duckdb-data/releases/download/v1.0/taxi_2019_04.parquet
```

## 1. Create a Context

We'll set up a `Context`:

```python
from gears import Gear, History, OpenAIChat
from pydantic import BaseModel
from typing import Any

import asyncio
import duckdb

class SQLContext(BaseModel):
    nlquery: str
    sql: str = None
    exception: str = None
    result: Any = None

table_statistics = duckdb.query("PRAGMA show_tables_expanded;").fetchdf()
system_prompt = f"Here's some statistics about my database:\n{table_statistics.to_string(index=False)}"
```

## 2. Create a Gear

We'll create a `Gear` that takes a natural language query and generates executable SQL:

```python
from gears.utils import extract_first_json

class SQLGear(Gear):
    def __init__(self, model: OpenAIChat, error: bool = False, **kwargs):
        self.error = error
        super().__init__(model, **kwargs)

    def template(self):
        if self.error:
            return "Your query failed to execute with the following error: {{ exception }}\n\nPlease try again. Output the SQL as a JSON with key `sql` and value equal to the SQL query for me to run."

        else:
            return "Translate the following query into SQL: {{ nlquery }}\n\nMake sure the SQL is executable. Output the SQL as a JSON with key `sql` and value equal to the SQL query for me to run."

    def transform(self, response: dict, context: SQLContext):
        reply = response["choices"][0]["message"]["content"].strip()

        # Get JSON from reply and execute it
        try:
            sql = extract_first_json(reply, context) # (1)!

            # Execute SQL
            result = duckdb.query(sql).fetchall()
            return SQLContext(nlquery=context.nlquery, sql=sql, result=result)

        except Exception as e:
            return SQLContext(nlquery=context.nlquery, exception=str(e))

    def switch(self, context: SQLContext):
        if context.exception is not None:
            return SQLGear(OpenAIChat("gpt-3.5-turbo"), error=True)
        else:
            return None
```

1. `extract_first_json` is a helper function that extracts the first JSON from a string. You can write your own parser here if you'd like.

## 3. Run the Gear

We'll run the gear like so:

```python
async def main():
    context = SQLContext(nlquery="How many trips were taken in April 2019?")
    history = History(system_message=system_prompt)
    llm = OpenAIChat("gpt-3.5-turbo")

    context = await SQLGear(llm).run(context, history)
    print(f"SQL query: {context.sql}")
    print(f"SQL query result: {context.result}")
    print(f"Cost of query: {history.cost}")
    print(history)

if __name__ == "__main__":
    asyncio.run(main())
```

This will run the gear until it returns `None` from `switch`, denoting a valid SQL query result. The output should look something like this:

```
TODO
```

### Extra Notes

`gears` is not a full-fledged LLM guardrails library, nor does it intend to be. It is just a lightweight way to add some validation criteria to your LLMs by using control flow. If you want to use a full-fledged LLM guardrails library, I recommend you check out [Guardrails](https://shreyar.github.io/guardrails/), coincidentally written by another person named Shreya.
