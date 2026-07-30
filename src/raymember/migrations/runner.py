"""Lightweight SQLite database migration runner with backup protection."""

import os
import shutil
import sqlite3
from typing import Tuple


class MigrationRunner:
    """Manages schema versioning, automatic backup creation, and non-destructive column additions."""

    CURRENT_SCHEMA_VERSION = 4

    def __init__(self, db_path: str):
        self.db_path = db_path

    def check_and_migrate(self) -> Tuple[int, int]:
        if not os.path.exists(self.db_path) or self.db_path == ":memory:":
            return self.CURRENT_SCHEMA_VERSION, self.CURRENT_SCHEMA_VERSION

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 1. Check or create schema_version table
        cursor.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, updated_at TEXT)")
        cursor.execute("SELECT MAX(version) FROM schema_version")
        row = cursor.fetchone()
        old_version = row[0] if (row and row[0] is not None) else 1

        if old_version >= self.CURRENT_SCHEMA_VERSION:
            conn.close()
            return old_version, old_version

        # 2. Backup database file before migration
        backup_path = f"{self.db_path}.v{old_version}.bak"
        shutil.copy2(self.db_path, backup_path)

        try:
            # v2 Migration: namespace and provenance columns
            tables = ["entities", "observations", "current_state", "state_transitions"]
            for tbl in tables:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (tbl,))
                if not cursor.fetchone():
                    continue

                cursor.execute(f"PRAGMA table_info({tbl})")
                cols = [c[1] for c in cursor.fetchall()]

                if "namespace" not in cols:
                    cursor.execute(f"ALTER TABLE {tbl} ADD COLUMN namespace TEXT DEFAULT 'default'")
                if "provenance" not in cols and tbl in ("observations", "current_state", "state_transitions"):
                    cursor.execute(f"ALTER TABLE {tbl} ADD COLUMN provenance TEXT DEFAULT 'sensor'")

            # v3 Migration: resolution metadata columns in observations & location_aliases table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='observations'")
            if cursor.fetchone():
                cursor.execute("PRAGMA table_info(observations)")
                obs_cols = [c[1] for c in cursor.fetchall()]

                if "raw_location" not in obs_cols:
                    cursor.execute("ALTER TABLE observations ADD COLUMN raw_location TEXT")
                if "normalized_location" not in obs_cols:
                    cursor.execute("ALTER TABLE observations ADD COLUMN normalized_location TEXT")
                if "canonical_location" not in obs_cols:
                    cursor.execute("ALTER TABLE observations ADD COLUMN canonical_location TEXT")
                if "resolution_method" not in obs_cols:
                    cursor.execute("ALTER TABLE observations ADD COLUMN resolution_method TEXT DEFAULT 'EXACT'")
                if "resolution_confidence" not in obs_cols:
                    cursor.execute("ALTER TABLE observations ADD COLUMN resolution_confidence REAL DEFAULT 1.0")
                if "resolution_confirmed" not in obs_cols:
                    cursor.execute("ALTER TABLE observations ADD COLUMN resolution_confirmed INTEGER DEFAULT 0")

            # v4 Migration: generalized state & attribute-level belief columns
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='observations'")
            if cursor.fetchone():
                cursor.execute("PRAGMA table_info(observations)")
                obs_cols = [c[1] for c in cursor.fetchall()]
                if "state_json" not in obs_cols:
                    cursor.execute("ALTER TABLE observations ADD COLUMN state_json TEXT DEFAULT '{}'")

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='current_state'")
            if cursor.fetchone():
                cursor.execute("PRAGMA table_info(current_state)")
                cs_cols = [c[1] for c in cursor.fetchall()]
                if "attributes_json" not in cs_cols:
                    cursor.execute("ALTER TABLE current_state ADD COLUMN attributes_json TEXT DEFAULT '{}'")
                if "attribute_beliefs_json" not in cs_cols:
                    cursor.execute("ALTER TABLE current_state ADD COLUMN attribute_beliefs_json TEXT DEFAULT '{}'")

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='state_transitions'")
            if cursor.fetchone():
                cursor.execute("PRAGMA table_info(state_transitions)")
                st_cols = [c[1] for c in cursor.fetchall()]
                if "attribute_name" not in st_cols:
                    cursor.execute("ALTER TABLE state_transitions ADD COLUMN attribute_name TEXT DEFAULT 'location'")
                if "old_value_json" not in st_cols:
                    cursor.execute("ALTER TABLE state_transitions ADD COLUMN old_value_json TEXT DEFAULT '{}'")
                if "new_value_json" not in st_cols:
                    cursor.execute("ALTER TABLE state_transitions ADD COLUMN new_value_json TEXT DEFAULT '{}'")

            # Create location_aliases table if not existing
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS location_aliases (
                    alias_id TEXT PRIMARY KEY,
                    raw_alias TEXT,
                    canonical_location TEXT,
                    namespace TEXT DEFAULT 'default',
                    status TEXT DEFAULT 'CONFIRMED',
                    provenance TEXT DEFAULT 'user_confirmed',
                    created_at TEXT
                )
            """)

            cursor.execute("INSERT OR REPLACE INTO schema_version (version, updated_at) VALUES (?, datetime('now'))", (self.CURRENT_SCHEMA_VERSION,))
            conn.commit()
            conn.close()
            return old_version, self.CURRENT_SCHEMA_VERSION

        except Exception as e:
            conn.rollback()
            conn.close()
            raise RuntimeError(f"Migration failed from version {old_version} to {self.CURRENT_SCHEMA_VERSION}: {str(e)}")
