"""Unit tests for Layer 1 storage persistence and ORM models."""

import os
import pytest
from raymember.schemas import Location, ObservationInput
from raymember.storage.database import DatabaseManager
from raymember.storage.models import (
    EntityRepository,
    ObservationRepository,
    CurrentStateRepository,
)


def test_entity_creation_and_retrieval(tmp_path):
    db_file = str(tmp_path / "test_storage.db")
    db_mgr = DatabaseManager(db_file)

    with db_mgr.session_scope() as session:
        repo = EntityRepository(session)
        ent = repo.create(canonical_name="backpack", entity_type="object", attributes={"color": "black", "owner": "Ray"})
        assert ent.entity_id.startswith("entity_")
        assert ent.canonical_name == "backpack"
        assert ent.attributes["color"] == "black"

        fetched = repo.get_by_id(ent.entity_id)
        assert fetched is not None
        assert fetched.canonical_name == "backpack"

    db_mgr.close()


def test_append_only_observation_storage(tmp_path):
    db_file = str(tmp_path / "test_obs.db")
    db_mgr = DatabaseManager(db_file)

    with db_mgr.session_scope() as session:
        e_repo = EntityRepository(session)
        o_repo = ObservationRepository(session)

        ent = e_repo.create("laptop", "device", {"brand": "Dell"})

        obs1 = ObservationInput(
            entity="laptop",
            location=Location(room="bedroom", x=1.0, y=2.0, z=0.5),
            confidence=0.9,
            source="camera",
        )
        obs_rec1 = o_repo.add(ent.entity_id, obs1)

        obs2 = ObservationInput(
            entity="laptop",
            location=Location(room="living_room", x=5.0, y=3.0, z=0.5),
            confidence=0.95,
            source="camera",
        )
        obs_rec2 = o_repo.add(ent.entity_id, obs2)

        history = o_repo.get_by_entity(ent.entity_id)
        assert len(history) == 2
        # History is ordered descending by timestamp
        assert history[0].observation_id == obs_rec2.observation_id
        assert history[1].observation_id == obs_rec1.observation_id

    db_mgr.close()


def test_sqlite_persistence_across_reopen(tmp_path):
    db_file = str(tmp_path / "persistent_test.db")

    # Step 1: Open, write entity & observation, then close
    db_mgr1 = DatabaseManager(db_file)
    with db_mgr1.session_scope() as session:
        e_repo = EntityRepository(session)
        o_repo = ObservationRepository(session)
        c_repo = CurrentStateRepository(session)

        ent = e_repo.create("keys", "keys", {"type": "house"})
        o_repo.add(ent.entity_id, ObservationInput(entity="keys", location=Location(room="kitchen"), confidence=0.85))
        c_repo.upsert(
            entity_id=ent.entity_id,
            room="kitchen",
            confidence=0.85,
            last_seen="2026-07-29T10:00:00Z",
            status="OBSERVED",
        )
    db_mgr1.close()

    # Step 2: Open new connection and verify records persist
    db_mgr2 = DatabaseManager(db_file)
    with db_mgr2.session_scope() as session:
        e_repo = EntityRepository(session)
        o_repo = ObservationRepository(session)
        c_repo = CurrentStateRepository(session)

        ents = e_repo.find_by_canonical_name("keys")
        assert len(ents) == 1
        eid = ents[0].entity_id

        obs_list = o_repo.get_by_entity(eid)
        assert len(obs_list) == 1
        assert obs_list[0].room == "kitchen"

        state = c_repo.get(eid)
        assert state is not None
        assert state.room == "kitchen"
        assert state.confidence == 0.85

    db_mgr2.close()
