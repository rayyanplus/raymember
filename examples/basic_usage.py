"""Basic usage example for Raymember WorldMemory."""

from raymember import WorldMemory

def main():
    print("=== Raymember Basic Usage Demo ===")

    memory = WorldMemory(database_path="raymember_basic.db")

    memory.observe(
        entity="laptop",
        attributes={"brand": "Dell", "owner": "Ray"},
        location={"room": "office", "x": 1.5, "y": 2.0, "z": 0.8},
        confidence=0.95,
        source="camera",
    )

    result = memory.query("Where is the laptop?")

    print(f"Answer: {result.answer}")
    print(f"Entity: {result.entity}")
    print(f"Current Location: {result.current_location}")
    print(f"Confidence: {result.confidence}")
    print(f"State: {result.state}")

    memory.close()

if __name__ == "__main__":
    main()
