# Text to Executable SQL (Simple Control Flow)

While many natural language to SQL systems these days can generate compilable SQL, it's hard for them to generate _executable_ SQL. For example, the following query is not executable if the `users` table does not have a `name` column:

```
SELECT * FROM users WHERE name = "Alice"
```

One of the areas where `gears` shines is the ability to validate LLM output and _then_ dynamically decide what to do. In this tutorial, we will build a simple `Gear` that generates executable SQL from a natural language query.

## 0. Downloads

Assuming you've already configured your `openai` keys, download the Python libraries we'll be using in this tutorial:

```bash
pip install duckdb
pip install pandas
```

We'll be using the NYC Taxicab dataset, described in [this DuckDB blog post](https://duckdb.org/2021/06/25/querying-parquet.html). Download it using `wget` like so:

```bash
wget https://github.com/cwida/duckdb-data/releases/download/v1.0/taxi_2019_04.parquet
```

## 1. Create a Context

We'll set up a `Context`:

```python linenums="1"
from gears import Gear, History, OpenAIChat
from gears.utils import extract_first_json
from pydantic import BaseModel
from typing import Any

import asyncio
import duckdb

class SQLContext(BaseModel):
    nlquery: str
    sql: str = None
    exception: str = None
    result: Any = None

# Import the NYC Taxi dataset
duckdb.query(
    "CREATE TABLE taxi_trips AS SELECT * FROM read_parquet('taxi_2019_04.parquet');"
)
table_statistics = duckdb.query("PRAGMA show_tables_expanded;").fetchdf()
table_statistics_str = str(table_statistics.iloc[0].to_dict())
system_prompt = (
    f"Here's some statistics about my database:\n{table_statistics_str}"
)
```

## 2. Create a Gear

We'll create a `Gear` that takes a natural language query and generates executable SQL:

```python linenums="24"
class SQLGear(Gear):
    def template(self, context: SQLContext):
        if context.exception:
            return "Your query failed to execute with the following error: {{ context.exception }}\n\nPlease try again. Output the SQL as a JSON with key `sql` and value equal to the SQL query for me to run."

        else:
            return "Translate the following query into SQL: {{ context.nlquery }}\n\nMake sure the SQL is executable. Output the SQL as a JSON with key `sql` and value equal to the SQL query for me to run."

    def transform(self, response: dict, context: SQLContext):
        reply = response["choices"][0]["message"]["content"].strip()

        # Get JSON from reply and execute it
        try:
            sql = extract_first_json(reply)["sql"] # (1)!

            # Execute SQL
            result = duckdb.query(sql).fetchall()
            return SQLContext(nlquery=context.nlquery, sql=sql, result=result)

        except Exception as e:
            return SQLContext(nlquery=context.nlquery, exception=str(e))

    def switch(self, context: SQLContext):
        if context.exception is not None:
            return SQLGear(OpenAIChat("gpt-3.5-turbo"))
        else:
            return None
```

1. `extract_first_json` is a helper function that extracts the first JSON from a string. You can write your own parser here if you'd like.

## 3. Run the Gear

We'll run the gear like so:

```python linenums="51"
async def main():
    context = SQLContext(nlquery="How many trips that cost more than $10 were taken in April 2019?")
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
SQL query: SELECT COUNT(*) FROM taxi_trips WHERE total_amount > 10 AND pickup_at >= '2019-04-01' AND pickup_at < '2019-05-01'
SQL query result: [(6281980,)]
Cost of query: 0.000462
[System]: Here's some statistics about my database:
{'database': 'memory', 'schema': 'main', 'name': 'taxi_trips', 'column_names': ['vendor_id', 'pickup_at', 'dropoff_at', 'passenger_count', 'trip_distance', 'rate_code_id', 'store_and_fwd_flag', 'pickup_location_id', 'dropoff_location_id', 'payment_type', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge', 'total_amount', 'congestion_surcharge'], 'column_types': ['VARCHAR', 'TIMESTAMP', 'TIMESTAMP', 'TINYINT', 'FLOAT', 'VARCHAR', 'VARCHAR', 'INTEGER', 'INTEGER', 'VARCHAR', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT', 'FLOAT'], 'temporary': False}
[User]: Translate the following query into SQL: How many trips that cost more than $10 were taken in April 2019?

Make sure the SQL is executable. Output the SQL as a JSON with key `sql` and value equal to the SQL query for me to run.
[Assistant]: {"sql": "SELECT COUNT(*) FROM taxi_trips WHERE total_amount > 10 AND pickup_at >= '2019-04-01' AND pickup_at < '2019-05-01'"}
```

## 4. Automatically Reprompt on Transform Failures

Sometimes, if you just ask the LLM the same question again (if you have a nonzero temperature, of course), the LLM will produce a correct or valid answer. To simply reprompt when your `transform` method raises some error (e.g., can't parse JSON or SQL), you can use the `num_retries_on_transform_error` parameter in a `Gear` constructor:

```python
gear = SQLGear(llm, num_retries_on_transform_error=1)
```

This will automatically retry the `transform` method once if it raises an error. By default, `num_retries_on_transform_error` is set to 0, meaning that the `transform` method will not be retried if it raises an error.

### Extra Notes

`gears` is not a full-fledged LLM guardrails library, nor does it intend to be. It just provides an interface to specify control flow, which can be used for lightweight validation of LLM outfits. If you want to use a full-fledged LLM guardrails library, I recommend you check out [Guardrails](https://shreyar.github.io/guardrails/), coincidentally written by another person named Shreya.
