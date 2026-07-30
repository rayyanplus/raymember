import os
from raymember import Raymember

db_path = "test_diag_loc.db"
if os.path.exists(db_path):
    os.remove(db_path)

mem = Raymember(database_path=db_path)
mem.observe("item_0", location={"room": "living_room"}, confidence=0.90, provenance="sensor")
mem.observe("item_0", location={"room": "kitchen"}, confidence=0.95, provenance="user")
mem.observe("item_0", location={"room": "garage"}, confidence=0.20, provenance="unreliable_sensor")

st = mem.get("item_0")
print("LOCATION CONF OBS:", st.conflicting_observations)
print("INTERPRETED HISTORY:", st.interpreted_history)

mem.close()
if os.path.exists(db_path):
    os.remove(db_path)
