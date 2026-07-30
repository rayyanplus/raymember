"""Deterministic entity resolution engine."""

from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy.orm import Session

from raymember.schemas import ObservationInput
from raymember.storage.models import EntityModel, EntityRepository


class EntityResolver:
    """Resolves incoming observations to existing entities or creates new ones."""

    def __init__(self, session: Session, namespace: str = "default"):
        self.session = session
        self.namespace = namespace
        self.entity_repo = EntityRepository(session, namespace=namespace)

    def resolve_or_create(self, obs_input: ObservationInput) -> Tuple[EntityModel, bool]:
        """
        Resolves observation to an Entity.
        Returns tuple: (EntityModel, is_newly_created).
        """
        if obs_input.entity_id:
            existing = self.entity_repo.get_by_id(obs_input.entity_id, namespace=self.namespace)
            if existing:
                if obs_input.attributes:
                    self.entity_repo.update_attributes(existing.entity_id, obs_input.attributes)
                return existing, False
            new_entity = self.entity_repo.create(
                canonical_name=obs_input.entity,
                entity_type=obs_input.entity_type or "object",
                attributes=obs_input.attributes,
                entity_id=obs_input.entity_id,
                namespace=self.namespace,
            )
            return new_entity, True

        label = obs_input.entity.strip()
        attrs = obs_input.attributes or {}

        by_name = self.entity_repo.find_by_canonical_name(label, namespace=self.namespace)

        matching_by_name = []
        for candidate in by_name:
            if self._attributes_compatible(candidate.attributes, attrs):
                matching_by_name.append(candidate)

        if len(matching_by_name) == 1:
            matched = matching_by_name[0]
            if attrs:
                self.entity_repo.update_attributes(matched.entity_id, attrs)
            return matched, False

        entity_type_candidates = self.entity_repo.find_by_type(obs_input.entity_type or label, namespace=self.namespace)
        strong_overlap_candidates = []
        for candidate in entity_type_candidates:
            if candidate in matching_by_name:
                continue
            if self._has_strong_attribute_overlap(candidate.attributes, attrs):
                strong_overlap_candidates.append(candidate)

        if len(strong_overlap_candidates) == 1:
            matched = strong_overlap_candidates[0]
            if attrs:
                self.entity_repo.update_attributes(matched.entity_id, attrs)
            return matched, False

        new_entity = self.entity_repo.create(
            canonical_name=label,
            entity_type=obs_input.entity_type or label,
            attributes=attrs,
            namespace=self.namespace,
        )
        return new_entity, True

    def _attributes_compatible(self, stored: Dict[str, Any], incoming: Dict[str, Any]) -> bool:
        if not stored or not incoming:
            return True
        for k, v in incoming.items():
            if k in stored and stored[k] != v:
                return False
        return True

    def _has_strong_attribute_overlap(self, stored: Dict[str, Any], incoming: Dict[str, Any]) -> bool:
        if not stored or not incoming:
            return False
        common_keys = set(stored.keys()).intersection(set(incoming.keys()))
        if not common_keys:
            return False
        matches = sum(1 for k in common_keys if stored[k] == incoming[k])
        return matches >= len(common_keys) / 2.0
