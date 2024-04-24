# Personalized Vacation Planner (Advanced Control Flow)

Sometimes we will want to generate a "tree" of calls to an LLM, where the next call depends on the output of the previous call. For example, suppose we want to generate a structured vacation itinerary for a user that includes activities that are dependent on the climate.

## 0. High-Level Overview

In this application, we'll use LLMs to:

- Based on a home location, pick a place to travel and season to travel there
- Select 2 activities to do at the destination, including outdoor and indoor activities (weather permitting)
- Summarize the activities into a travel itinerary

## 1. Create a Context

We'll set up a Pydantic model with information we'll extract from the LLM:

```python linenums="1"
from gears import Gear, History, OpenAIChat
from gears.utils import extract_first_json
from pydantic import BaseModel
from typing import Any

import asyncio

class Suggestion(BaseModel, extra="allow"):
    home: str
    destination: str = None
    season: str = None
    activities: list = None
    itinerary: str = None
```

## 2. Create a Gear to Select a Destination/Season and Outdoor/Indoor Activities

We'll create a `Gear` that takes a home location and generates a destination and season to travel there:

```python linenums="14"
class DestinationSelection(Gear):
    def prompt(self, context: Suggestion):
        return "Suggest a city within a 4-5 hour flight that someone who lives in {{ context.home }} can travel to for a vacation. Pick a season that is best to travel to this destination in. Output the destination and season as a JSON with keys `destination` and `season` and values equal to the destination and season, respectively."

    def transform(self, response: dict, context: Suggestion):
        reply = response["choices"][0]["message"]["content"].strip()

        # Get JSON from reply and execute it
        answer = extract_first_json(reply)
        return Suggestion(
            home=context.home,
            destination=answer["destination"],
            season=answer["season"],
        )

    def switch(self, context: Suggestion):
        return IndoorOrOutdoor(OpenAIChat("gpt-3.5-turbo"))


class IndoorOrOutdoor(Gear):
    def prompt(self, context: Suggestion):
        return "Based on the expected weather at the destination in the {{ context.season }} season, can the person do outdoor activities? Output your answer as a JSON with key `outdoor` and value equal to `yes` or `no`, respectively."

    def transform(self, response: dict, context: Suggestion):
        reply = response["choices"][0]["message"]["content"].strip()

        # Get JSON from reply and execute it
        answer = extract_first_json(reply)
        return Suggestion(
            home=context.home,
            destination=context.destination,
            season=context.season,
            outdoor=answer["outdoor"],
        )

    def switch(self, context: Suggestion):
        if context.outdoor == "yes":
            return OutdoorActivitySelection(OpenAIChat("gpt-3.5-turbo"))
        else:
            return IndoorActivitySelection(OpenAIChat("gpt-3.5-turbo"))
```

## 3. Create Gears to Select the Activities

Now, we'll ask the LLM to select activities to do at the destination:

```python linenums="54"
class OutdoorActivitySelection(Gear):
    def prompt(self, context: Suggestion):
        return "Suggest a popular outdoor activity to do in the city of {{ context.destination }} during the season of {{ context.season }}."

    def transform(self, response: dict, context: Suggestion):
        reply = response["choices"][0]["message"]["content"].strip()

        activities = [reply]
        return Suggestion(
            home=context.home,
            destination=context.destination,
            season=context.season,
            activities=activities,
            outdoor=context.outdoor,
        )

    def switch(self, context: Suggestion):
        return IndoorActivitySelection(OpenAIChat("gpt-3.5-turbo"))


class IndoorActivitySelection(Gear):
    def prompt(self, context: Suggestion):
        # If there is already an indoor activity, prompt for a different one
        if context.activities:
            return "Suggest a different popular indoor activity to do in the city of {{ context.destination }} during the season of {{ context.season }}."
        else:
            return "Suggest a popular indoor activity to do in the city of {{ context.destination }} during the season of {{ context.season }}."

    def transform(self, response: dict, context: Suggestion):
        reply = response["choices"][0]["message"]["content"].strip()

        activities = (
            context.activities + [reply] if context.activities else [reply]
        )
        return Suggestion(
            home=context.home,
            destination=context.destination,
            season=context.season,
            activities=activities,
            outdoor=context.outdoor,
        )

    def switch(self, context: Suggestion):
        if context.outdoor != "yes":
            # Need another indoor activity
            return IndoorActivitySelection(OpenAIChat("gpt-3.5-turbo"))
        else:
            # Already have 2 activities. Go to summary
            return SummarizeItinerary(OpenAIChat("gpt-3.5-turbo"))
```

Note that we have a special case in the `IndoorActivitySelection` gear: if the LLM recommended not to do outdoor activities at all, then we ask for 2 indoor activities. Otherwise, we ask for 1 indoor activity and 1 outdoor activity.

## 4. Create a Gear to Summarize the Itinerary

Finally, we'll ask the LLM to summarize the itinerary:

```python linenums="103"
class SummarizeItinerary(Gear):
    def prompt(self, context: Suggestion):
        return "Summarize your suggested activities: {{ context.activities }} into a short personalized vacation itinerary for someone who lives in {{ context.home }} to travel to {{ context.destination }} during the {{ context.season }} season."

    def transform(self, response: dict, context: Suggestion):
        reply = response["choices"][0]["message"]["content"].strip()

        return Suggestion(
            home=context.home,
            destination=context.destination,
            season=context.season,
            activities=context.activities,
            outdoor=context.outdoor,
            itinerary=reply,
        )
```

The `switch` method is not implemented, so the workflow will end after this gear.

## 5. Run the Workflow

We can run the workflow for a user living in San Francisco as follows:

```python linenums="118"
async def main():
    context = Suggestion(
        home="San Francisco, CA",
    )
    history = History(system_message="You are a professional travel agent.")
    llm = OpenAIChat("gpt-3.5-turbo")

    context = await DestinationSelection(llm).run(context, history)
    print(f"Resulting itinerary: {context.itinerary}")
    print(f"Cost of using the LLM: {history.cost}")
    # (1)!


if __name__ == "__main__":
    asyncio.run(main())
```

1. You can also print the history to see the full conversation, i.e., `print(history)`

The output should look something like this:

```
Resulting itinerary: Here's a personalized vacation itinerary for your trip to New York City, NY during the Autumn season:

Day 1:
- Arrive in New York City and settle into your accommodation.
- Take a stroll through Central Park and enjoy the beautiful autumn foliage.

Day 2:
- Explore the iconic landmarks of New York City, such as Times Square, Empire State Building, and Statue of Liberty.
- In the evening, indulge in a fantastic Broadway show.

Day 3:
- Visit renowned museums like the Metropolitan Museum of Art or the Museum of Modern Art.
- Discover the vibrant neighborhoods of Manhattan, such as SoHo, Greenwich Village, or Chelsea.

Day 4:
- Take a leisurely walk along the High Line, a beautiful elevated park with stunning views of the city.
- Explore the trendy shops and boutiques in neighborhoods like Fifth Avenue or Madison Avenue.

Day 5:
- Experience the bustling local markets, such as Union Square Greenmarket or Chelsea Market.
- Enjoy a delicious brunch and immerse yourself in the city's diverse food scene.

Day 6:
- Final day to explore any missed attractions, do some souvenir shopping, or relax in a cozy café.
- Depart from New York City and return home with wonderful memories.

This itinerary combines outdoor activities like strolling in Central Park with indoor excitement like watching a Broadway show, ensuring you make the most of your trip to New York City during the beautiful autumn season.
Cost of using the LLM: 0.0022164999999999997
```
