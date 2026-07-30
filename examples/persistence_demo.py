"""Process restart and SQLite persistence demo."""

import os
from raymember import WorldMemory

def main():
    db_filename = "raymember_persistence_demo.db"

    # Clean previous run file if any
    if os.path.exists(db_filename):
        os.remove(db_filename)

    print("=== Raymember Persistence Demo ===")

    print("\n1. Session A: Writing memory & closing process...")
    mem1 = WorldMemory(database_path=db_filename)
    mem1.observe(
        entity="house_keys",
        attributes={"owner": "Ray"},
        location={"room": "hallway", "x": 0.5, "y": 1.2, "z": 0.9},
        confidence=0.98,
    )
    mem1.close()
    print("   Session A closed.")

    print("\n2. Session B: Reopening database from disk...")
    mem2 = WorldMemory(database_path=db_filename)
    res = mem2.query("Where are the house_keys?")

    print(f"   Query Answer: {res.answer}")
    print(f"   Current Location: {res.current_location}")
    print(f"   Confidence: {res.confidence}")
    mem2.close()
    print("\n   Persistence verified successfully!")

if __name__ == "__main__":
    main()
