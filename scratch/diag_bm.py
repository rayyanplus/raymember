import os
from raymember import Raymember

db_path = "test_diag_bm.db"
if os.path.exists(db_path):
    os.remove(db_path)

mem = Raymember(database_path=db_path)
mem.observe("task_0", state={"owner": "agent_alpha"}, confidence=0.95, provenance="user")
print("AFTER STEP 1:", mem.get("task_0").attribute_beliefs)

mem.observe("task_0", state={"owner": "agent_beta"}, confidence=0.40, provenance="agent")
print("AFTER STEP 2:", mem.get("task_0").attribute_beliefs)

mem.close()
if os.path.exists(db_path):
    os.remove(db_path)
