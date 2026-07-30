from raymember import Raymember

memory = Raymember(
    database_path="my_memory.db"
)

# The backpack starts in the bedroom
memory.observe(
    entity="black backpack",
    location={
        "room": "bedroom"
    },
    confidence=0.95,
    provenance="user"
)

print("\nQUESTION 1")
print(
    memory.ask(
        "Where is the black backpack?"
    )
)

# The backpack is moved
memory.observe(
    entity="black backpack",
    location={
        "room": "living room"
    },
    confidence=0.95,
    provenance="user"
)

print("\nQUESTION 2")
print(
    memory.ask(
        "Where is the black backpack?"
    )
)

print("\nQUESTION 3")
print(
    memory.ask(
        "Where was the black backpack before?"
    )
)

print("\nMEMORY HISTORY")
print(
    memory.history(
        "black backpack"
    )
)

memory.close()