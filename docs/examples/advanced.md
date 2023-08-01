# Personalized Vacation Planner (Advanced Control Flow)

Sometimes we will want to generate a "tree" of calls to an LLM, where the next call depends on the output of the previous call. For example, suppose we want to generate a structured vacation itinerary for a user that includes activities that are dependent on the climate.

## 0. High-Level Overview

In this application, we'll use LLMs to:

- Based on a home location, pick a place to travel and season to travel there
- Select 2 activities to do at the destination, including outdoor and indoor activities (weather permitting)
- Summarize the activities into a travel itinerary

## 1. Create a Context

We'll set up a Pydantic model with information we'll extract from the LLM:

```python
from gears import Gear, History, OpenAIChat
from pydantic import BaseModel
from typing import Any

import asyncio

class Suggestion(BaseModel):
    home: str
    destination: str = None
    season: str = None
    activities: list = None
    itinerary: str = None
```

## 2. Create a Gear to Select a Destination/Season and Outdoor/Indoor Activities

We'll create a `Gear` that takes a home location and generates a destination and season to travel there:

```python
from gears.utils import extract_first_json

class DestinationSelection(Gear):
    def template(self, context: Suggestion):
        return "Suggest a city within a 4-5 hour flight that someone who lives in {{ context.home }} can travel to for a vacation. Pick a season that is best to travel to this destination in. Output the destination and season as a JSON with keys `destination` and `season` and values equal to the destination and season, respectively."

    def transform(self, response: dict, context: Suggestion):
        reply = response["choices"][0]["message"]["content"].strip()

        # Get JSON from reply and execute it
        answer = extract_first_json(reply, context)
        return Suggestion(home=context.home, destination=answer["destination"], season=answer["season"])

    def switch(self, context: Suggestion):
        return IndoorOrOutdoor(OpenAIChat("gpt-3.5-turbo"))

class IndoorOrOutdoor(Gear):
    def template(self, context: Suggestion):
        return "Based on the expected weather at the destination in the {{ context.season }} season, can the person do outdoor activities? Output your answer as a JSON with key `outdoor` and value equal to `yes` or `no`, respectively."

    def transform(self, response: dict, context: Suggestion):
        reply = response["choices"][0]["message"]["content"].strip()

        # Get JSON from reply and execute it
        answer = extract_first_json(reply, context)
        return Suggestion(home=context.home, destination=answer["destination"], season=answer["season"], outdoor=answer["outdoor"])

    def switch(self, context: Suggestion):
        if context.outdoor == "yes":
            return OutdoorActivitySelection(OpenAIChat("gpt-3.5-turbo"))
        else:
            return IndoorActivitySelection(OpenAIChat("gpt-3.5-turbo"))
```

## 3. Create Gears to Select the Activities

Now, we'll ask the LLM to select activities to do at the destination:

```python
class OutdoorActivitySelection(Gear):
    def template(self, context: Suggestion):
        return "Suggest a popular outdoor activity to do in the city of {{ context.destination }} during the season of {{ context.season }}."

    def transform(self, response: dict, context: Suggestion):
        reply = response["choices"][0]["message"]["content"].strip()

        activities = [reply]
        return Suggestion(home=context.home, destination=context.destination, season=context.season, activities=activities, outdoor=context.outdoor)

    def switch(self, context: Suggestion):
        return IndoorOrOutdoor(OpenAIChat("gpt-3.5-turbo"))

class IndoorActivitySelection(Gear):
    def template(self, context: Suggestion):
        # If there is already an indoor activity, prompt for a different one
        if context.activities:
            return "Suggest a different popular indoor activity to do in the city of {{ context.destination }} during the season of {{ context.season }}."
        else:
            return "Suggest a popular indoor activity to do in the city of {{ context.destination }} during the season of {{ context.season }}."

    def transform(self, response: dict, context: Suggestion):
       reply = response["choices"][0]["message"]["content"].strip()

        activities = context.activities + [reply] if context.activities else [reply]
        return Suggestion(home=context.home, destination=context.destination, season=context.season, activities=activities, outdoor=context.outdoor)

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

```python
class SummarizeItinerary(Gear):
    def template(self, context: Suggestion):
        return "Summarize your suggested activities: {{ context.activities }} into a short personalized vacation itinerary for someone who lives in {{ context.home }} to travel to {{ context.destination }} during the {{ context.season }} season."

    def transform(self, response: dict, context: Suggestion):
        reply = response["choices"][0]["message"]["content"].strip()

        return Suggestion(home=context.home, destination=context.destination, season=context.season, activities=activities, outdoor=context.outdoor, itinerary=reply)
```

The `switch` method is not implemented, so the workflow will end after this gear.

## 5. Run the Workflow

TODO
