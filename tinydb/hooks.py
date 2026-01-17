"""
This module implements a hook system for TinyDB that allows users to
register callbacks for database events.

The hook system supports both "before" and "after" events for insert,
update, and delete operations. Hooks are designed to be non-intrusive:
they cannot modify documents or database state, and errors in hooks
do not interrupt database operations.
"""

from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Union
import warnings


class HookEvent(Enum):
    """
    Enumeration of all supported hook events.

    Events are organized by operation type (insert, update, delete) and
    timing (before or after the operation).
    """
    # Insert events
    BEFORE_INSERT = 'before_insert'
    AFTER_INSERT = 'after_insert'

    # Update events
    BEFORE_UPDATE = 'before_update'
    AFTER_UPDATE = 'after_update'

    # Delete events (includes remove, truncate, soft_remove, restore, purge)
    BEFORE_DELETE = 'before_delete'
    AFTER_DELETE = 'after_delete'


# Type alias for hook callbacks
# Callbacks receive: (table_name: str, event: HookEvent, documents: List[Dict])
# 'documents' contains the affected document(s) with their doc_ids
HookCallback = Callable[[str, HookEvent, List[Dict[str, Any]]], None]


class HookManager:
    """
    Manages hook registration and execution for database events.

    The HookManager allows users to register callbacks that are executed
    when specific database events occur. Hooks are non-blocking and
    error-safe: if a hook fails, the error is caught and a warning is
    issued, but the database operation continues normally.

    Usage example:

    >>> from tinydb import TinyDB
    >>> from tinydb.hooks import HookEvent
    >>>
    >>> def on_insert(table_name, event, documents):
    ...     print(f"Inserted into {table_name}: {documents}")
    ...
    >>> db = TinyDB('db.json')
    >>> db.hooks.register(HookEvent.AFTER_INSERT, on_insert)
    >>> db.insert({'name': 'Alice'})
    Inserted into _default: [{'name': 'Alice', 'doc_id': 1}]
    """

    def __init__(self):
        """
        Initialize the hook manager with empty hook registries.
        """
        self._hooks: Dict[HookEvent, List[HookCallback]] = {
            event: [] for event in HookEvent
        }

    def register(
        self,
        event: HookEvent,
        callback: HookCallback
    ) -> 'HookManager':
        """
        Register a callback for a specific event.

        :param event: The event to register the callback for
        :param callback: The callback function to execute when the event occurs.
                        The callback receives three arguments:
                        - table_name (str): The name of the affected table
                        - event (HookEvent): The event type
                        - documents (List[Dict]): The affected documents with
                          their doc_ids included as 'doc_id' key
        :returns: The HookManager instance for method chaining

        Example:

        >>> def my_callback(table_name, event, documents):
        ...     for doc in documents:
        ...         print(f"Doc {doc['doc_id']} affected in {table_name}")
        ...
        >>> manager.register(HookEvent.AFTER_INSERT, my_callback)
        """
        if not isinstance(event, HookEvent):
            raise ValueError(f"Invalid event type: {event}. "
                           f"Must be a HookEvent enum value.")

        if not callable(callback):
            raise ValueError("Callback must be callable")

        self._hooks[event].append(callback)
        return self

    def unregister(
        self,
        event: HookEvent,
        callback: HookCallback
    ) -> bool:
        """
        Unregister a callback from a specific event.

        :param event: The event to unregister the callback from
        :param callback: The callback function to remove
        :returns: True if the callback was found and removed, False otherwise
        """
        if not isinstance(event, HookEvent):
            raise ValueError(f"Invalid event type: {event}. "
                           f"Must be a HookEvent enum value.")

        try:
            self._hooks[event].remove(callback)
            return True
        except ValueError:
            return False

    def unregister_all(self, event: Optional[HookEvent] = None) -> None:
        """
        Unregister all callbacks, optionally for a specific event.

        :param event: If provided, only unregister callbacks for this event.
                     If None, unregister all callbacks for all events.
        """
        if event is not None:
            if not isinstance(event, HookEvent):
                raise ValueError(f"Invalid event type: {event}. "
                               f"Must be a HookEvent enum value.")
            self._hooks[event] = []
        else:
            for e in HookEvent:
                self._hooks[e] = []

    def run(
        self,
        event: HookEvent,
        table_name: str,
        documents: List[Dict[str, Any]]
    ) -> None:
        """
        Execute all callbacks registered for a specific event.

        This method is called internally by TinyDB when database events occur.
        Errors in callbacks are caught and reported as warnings to prevent
        them from interrupting database operations.

        :param event: The event that occurred
        :param table_name: The name of the affected table
        :param documents: The affected documents (copies, not originals)
        """
        for callback in self._hooks[event]:
            try:
                callback(table_name, event, documents)
            except Exception as e:
                # Issue a warning but don't interrupt database operations
                warnings.warn(
                    f"Hook callback {callback.__name__ if hasattr(callback, '__name__') else callback} "
                    f"raised an exception for event {event.value}: {e}",
                    category=UserWarning,
                    stacklevel=2
                )

    def has_hooks(self, event: HookEvent) -> bool:
        """
        Check if any hooks are registered for a specific event.

        This can be used to avoid preparing hook data when no hooks
        are registered.

        :param event: The event to check
        :returns: True if at least one hook is registered for the event
        """
        return len(self._hooks[event]) > 0

    def get_hooks(self, event: HookEvent) -> List[HookCallback]:
        """
        Get a list of all hooks registered for a specific event.

        :param event: The event to get hooks for
        :returns: A list of callback functions (copy of the internal list)
        """
        return self._hooks[event][:]


__all__ = ('HookEvent', 'HookCallback', 'HookManager')
