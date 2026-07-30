from raymember import Raymember


# Create/open a separate database for this test
memory = Raymember(database_path="conflict_test.db")


# -------------------------------------------------
# OBSERVATION 1:
# User says the car keys are on the desk.
# High-trust observation.
# -------------------------------------------------

memory.observe(
    entity="car keys",
    location={"room": "desk"},
    confidence=0.98,
    provenance="user"
)


# -------------------------------------------------
# OBSERVATION 2:
# An agent claims the keys are in the kitchen.
# Lower-trust observation.
# -------------------------------------------------

memory.observe(
    entity="car keys",
    location={"room": "kitchen"},
    confidence=0.60,
    provenance="agent"
)


# -------------------------------------------------
# GET STRUCTURED CURRENT STATE
# -------------------------------------------------

result = memory.get("car keys")

print("\nCURRENT MEMORY OBJECT")
print(result)


# -------------------------------------------------
# PRINT A CLEAN HUMAN-READABLE RESULT
# -------------------------------------------------

print("\nCLEAN ANSWER")

print(f"Entity: {result.entity_label}")

print(
    "Current location:",
    result.current_location.get("room", "unknown")
)

print(f"Confidence: {result.confidence:.0%}")

print(f"State: {result.state}")

print(f"Uncertainty status: {result.uncertainty_status}")

print(f"Current provenance: {result.provenance}")

print(f"Explanation: {result.explanation}")


# -------------------------------------------------
# ASK A NATURAL-LANGUAGE QUESTION
# ask() returns a result containing an answer field.
# -------------------------------------------------

question_result = memory.ask(
    "Where are the car keys?"
)

print("\nNATURAL-LANGUAGE ANSWER")

# Different Raymember versions may return either:
# 1. an object with .answer
# 2. a plain string
if hasattr(question_result, "answer"):
    print(question_result.answer)
else:
    print(question_result)


# -------------------------------------------------
# SHOW THE COMPLETE HISTORY
# -------------------------------------------------

print("\nMEMORY HISTORY")

history = memory.history("car keys")

for index, observation in enumerate(history, start=1):

    location = observation.get(
        "location",
        {}
    )

    room = location.get(
        "room",
        observation.get("room", "unknown")
    )

    print(f"\nObservation {index}")

    print(f"Location: {room}")

    print(
        "Confidence:",
        observation.get("confidence")
    )

    print(
        "Provenance:",
        observation.get("provenance")
    )

    print(
        "Timestamp:",
        observation.get("timestamp")
    )


# -------------------------------------------------
# CLOSE THE DATABASE
# -------------------------------------------------

memory.close()