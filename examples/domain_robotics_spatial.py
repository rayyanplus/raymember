"""
Raymember Phase 5 Domain Example: Robotics & Spatial Object Location
Demonstrates backward-compatible 3D spatial object tracking and relocation.
"""

import os
from raymember import Raymember


def main():
    db_path = "robotics_demo.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    print("============================================================")
    print("  Raymember Phase 5 - Robotics & Spatial Object Demo")
    print("============================================================")

    mem = Raymember(database_path=db_path)

    # 1. Robot camera observes backpack in living_room
    print("\n--- Step 1: Vision camera observes backpack in living_room ---")
    mem.observe(
        entity="backpack",
        location={"room": "living_room", "x": 1.2, "y": 3.4, "z": 0.5},
        confidence=0.95,
        provenance="sensor",
    )

    state1 = mem.get("backpack")
    print(f"Current Location: {state1.current_location}")

    # 2. Relocation to office
    print("\n--- Step 2: Robot moves backpack to office ---")
    mem.observe(
        entity="backpack",
        location={"room": "office", "x": 5.0, "y": 1.2, "z": 0.8},
        confidence=0.98,
        provenance="sensor",
    )

    state2 = mem.get("backpack")
    print(f"Updated Location:  {state2.current_location}")
    print(f"Previous Location: {state2.previous_location}")

    # 3. Natural language queries
    q1 = mem.ask("Where is the backpack?")
    print(f"\nQuery:  'Where is the backpack?'")
    print(f"Answer: {q1.answer}")

    q2 = mem.ask("Where was the backpack before?")
    print(f"\nQuery:  'Where was the backpack before?'")
    print(f"Answer: {q2.answer}")

    mem.close()
    if os.path.exists(db_path):
        os.remove(db_path)
    print("\nRobotics spatial demo complete!")


if __name__ == "__main__":
    main()
