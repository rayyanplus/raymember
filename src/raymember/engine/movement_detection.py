"""Movement detection logic for 3D coordinates and room deltas."""

from typing import Any, Dict, Tuple, Union
from raymember.schemas import Location


class MovementDetector:
    """Detects spatial movement between current belief and new observation."""

    def __init__(self, movement_distance_threshold: float = 0.5):
        self.movement_distance_threshold = movement_distance_threshold

    def is_moved(
        self,
        old_location: Dict[str, Any],
        new_location: Union[Location, Dict[str, Any]],
    ) -> Tuple[bool, float, str]:
        """
        Determines whether an entity has moved.
        Returns tuple: (is_moved: bool, spatial_distance: float, reason: str).
        """
        if isinstance(new_location, Location):
            new_room = new_location.room
            new_x, new_y, new_z = new_location.x, new_location.y, new_location.z
        elif isinstance(new_location, dict):
            new_room = new_location.get("room")
            new_x, new_y, new_z = new_location.get("x"), new_location.get("y"), new_location.get("z")
        else:
            new_room = str(new_location)
            new_x, new_y, new_z = None, None, None

        old_room = old_location.get("room") if isinstance(old_location, dict) else str(old_location)
        old_x = old_location.get("x") if isinstance(old_location, dict) else None
        old_y = old_location.get("y") if isinstance(old_location, dict) else None
        old_z = old_location.get("z") if isinstance(old_location, dict) else None

        # Check for room change
        if old_room and new_room and str(old_room).lower() != str(new_room).lower():
            return True, 1.0, f"Room changed from '{old_room}' to '{new_room}'"

        # Check 3D Euclidean distance if coordinates are present in both
        if (
            old_x is not None
            and old_y is not None
            and old_z is not None
            and new_x is not None
            and new_y is not None
            and new_z is not None
        ):
            dist = ((new_x - old_x) ** 2 + (new_y - old_y) ** 2 + (new_z - old_z) ** 2) ** 0.5
            if dist > self.movement_distance_threshold:
                return True, dist, f"3D displacement of {dist:.2f}m exceeds threshold ({self.movement_distance_threshold}m)"
            return False, dist, f"Stationary (displacement {dist:.2f}m)"

        return False, 0.0, "Stationary (same room, insufficient 3D coordinates)"
