"""Object relocation demonstration matching primary target MVP specification."""

from raymember import WorldMemory

def main():
    print("=== Raymember Object Relocation Demo ===")

    memory = WorldMemory(database_path="raymember_relocation.db")

    print("\n[Step 1] Observing backpack in bedroom...")
    memory.observe(
        entity="backpack",
        attributes={
            "color": "black",
            "owner": "Ray"
        },
        location={
            "room": "bedroom",
            "x": 2.1,
            "y": 0.0,
            "z": 4.3
        },
        confidence=0.91,
        source="simulator"
    )

    print("[Step 2] Observing backpack in living_room...")
    memory.observe(
        entity="backpack",
        attributes={
            "color": "black",
            "owner": "Ray"
        },
        location={
            "room": "living_room",
            "x": 6.2,
            "y": 0.0,
            "z": 3.1
        },
        confidence=0.94,
        source="simulator"
    )

    print("\n[Step 3] Querying memory: 'Where is Ray\\'s black backpack?'")
    result = memory.query("Where is Ray's black backpack?")

    print("\nResult Answer:")
    print(result.answer)

    print("\nStructured Result Fields:")
    print(f"  entity: {result.entity}")
    print(f"  current_location: {result.current_location}")
    print(f"  confidence: {result.confidence}")
    print(f"  last_seen: {result.last_seen}")
    print(f"  previous_location: {result.previous_location}")
    print(f"  state: {result.state}")
    print(f"  evidence: {result.evidence}")

    print("\nFormatted Model-Agnostic LLM Context Export:")
    context = memory.get_context("Where is Ray's black backpack?")
    print(context.to_formatted_prompt())

    memory.close()

if __name__ == "__main__":
    main()
