"""Custom exceptions for Raymember."""


class RaymemberError(Exception):
    """Base class for all Raymember errors."""
    pass


class EntityNotFoundError(RaymemberError):
    """Raised when an requested entity is not found in memory."""
    pass


class AmbiguousEntityError(RaymemberError):
    """Raised when an entity reference matches multiple candidates ambiguously."""
    pass


class InvalidObservationError(RaymemberError):
    """Raised when observation input data is malformed or invalid."""
    pass


class QueryParseError(RaymemberError):
    """Raised when a natural language query cannot be parsed by the retrieval engine."""
    pass


class DatabaseError(RaymemberError):
    """Raised when a storage layer operation fails."""
    pass


class PolicyNotTrainedError(RaymemberError):
    """Raised when attempting to use an uninitialized or untrained ML update policy."""
    pass
