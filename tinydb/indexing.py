"""
This module implements field-level indexing for TinyDB tables.

Indexes provide fast lookups for queries on specific fields, avoiding
full table scans for equality and comparison operations.

Example usage:

>>> from tinydb import TinyDB, where
>>> db = TinyDB('db.json')
>>> table = db.table('users')
>>>
>>> # Create an index on the 'user_id' field
>>> table.create_index('user_id')
>>>
>>> # Queries on indexed fields are now much faster
>>> table.search(where('user_id') == 'user123')
>>>
>>> # List all indexes
>>> table.list_indexes()
['user_id']
>>>
>>> # Remove an index
>>> table.drop_index('user_id')
"""

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
from collections import defaultdict

__all__ = ('Index', 'IndexManager', 'UnhashableValueError')


class UnhashableValueError(Exception):
    """
    Raised when a field value cannot be converted to a hashable form for indexing.

    This exception is raised when attempting to index a document whose field
    value contains types that cannot be made hashable (e.g., custom objects
    without __hash__ defined).

    Attributes:
        field_name: The indexed field name (e.g., 'user.profile')
        value: The unhashable value that caused the error
        doc_id: The document ID containing the unhashable value
        value_path: The path within the field value where the unhashable
                   value was found (e.g., ['items', 2, 'data'] means the
                   unhashable value is at field_value['items'][2]['data'])
    """

    def __init__(
        self,
        field_name: str,
        value: Any,
        doc_id: int,
        value_path: Optional[List[Union[str, int]]] = None
    ):
        self.field_name = field_name
        self.value = value
        self.doc_id = doc_id
        self.value_path = value_path or []

        # Build detailed location message
        if self.value_path:
            path_str = self._format_path(self.value_path)
            location_msg = (
                f"field '{field_name}' at path {path_str} contains "
                f"unhashable value of type {type(value).__name__}"
            )
        else:
            location_msg = (
                f"field '{field_name}' contains "
                f"unhashable value of type {type(value).__name__}"
            )

        super().__init__(
            f"Cannot index document {doc_id}: {location_msg}. "
            f"Only hashable types (str, int, float, bool, None, list, dict) "
            f"are supported for indexing."
        )

    @staticmethod
    def _format_path(path: List[Union[str, int]]) -> str:
        """
        Format a path list into a readable string.

        Examples:
            [] -> ''
            ['key'] -> "['key']"
            [0] -> '[0]'
            ['items', 2, 'data'] -> "['items'][2]['data']"
        """
        if not path:
            return ''
        parts = []
        for p in path:
            if isinstance(p, int):
                parts.append(f'[{p}]')
            else:
                parts.append(f"['{p}']")
        return ''.join(parts)


class _UnhashableError(Exception):
    """
    Internal exception used to propagate unhashable value info with path.

    This is not part of the public API - it's caught by Index.add() and
    converted to UnhashableValueError with full context.
    """

    def __init__(self, value: Any, path: List[Union[str, int]]):
        self.value = value
        self.path = path
        super().__init__()


class Index:
    """
    A field index that maps field values to document IDs.

    This index supports:
    - Equality lookups: O(1) average case
    - Range queries (>, <, >=, <=): O(n) where n is number of unique values

    The index maintains a mapping from field values to sets of document IDs
    that have that value for the indexed field.
    """

    def __init__(self, field_name: str):
        """
        Create a new index for a field.

        :param field_name: The name of the field to index. Supports nested
                          fields using dot notation (e.g., 'address.city').
        """
        self._field_name = field_name
        self._field_path = field_name.split('.')
        # Maps field values to sets of document IDs
        self._value_to_doc_ids: Dict[Any, Set[int]] = defaultdict(set)
        # Maps document IDs to their indexed value (for updates/deletes)
        self._doc_id_to_value: Dict[int, Any] = {}

    @property
    def field_name(self) -> str:
        """Get the name of the indexed field."""
        return self._field_name

    def _get_field_value(self, document: Dict) -> Tuple[bool, Any]:
        """
        Extract the field value from a document.

        :param document: The document to extract the value from
        :returns: A tuple (found, value) where found indicates if the field
                 exists and value is the field value (or None if not found)
        """
        value = document
        for part in self._field_path:
            if not isinstance(value, dict) or part not in value:
                return (False, None)
            value = value[part]
        return (True, value)

    def _make_hashable(
        self,
        value: Any,
        path: Optional[List[Union[str, int]]] = None
    ) -> Any:
        """
        Convert a value to a hashable form for use as a dict key.

        Lists are converted to tuples, dicts to frozensets of items.
        Other values are returned as-is if hashable.

        :param value: The value to convert
        :param path: Current path within the value structure for error reporting
        :returns: A hashable version of the value
        :raises _UnhashableError: Internal error with path info if value can't be hashed
        """
        if path is None:
            path = []

        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            # Recursively convert list/tuple elements, tracking index in path
            result = []
            for i, v in enumerate(value):
                result.append(self._make_hashable(v, path + [i]))
            return tuple(result)
        if isinstance(value, dict):
            # Convert dict to frozenset of (key, hashable_value) pairs
            items = []
            for k, v in value.items():
                items.append((k, self._make_hashable(v, path + [k])))
            return frozenset(items)
        # For other types, try to hash directly
        try:
            hash(value)
            return value
        except TypeError:
            # Raise internal error with path information
            raise _UnhashableError(value, path)

    def add(self, doc_id: int, document: Dict) -> None:
        """
        Add a document to the index.

        If the field value cannot be made hashable, the document is skipped
        (not indexed) and an UnhashableValueError is raised with detailed
        path information showing exactly where the unhashable value is located.

        :param doc_id: The document ID
        :param document: The document data
        :raises UnhashableValueError: If the field value cannot be hashed
        """
        found, value = self._get_field_value(document)
        if found:
            try:
                hashable_value = self._make_hashable(value)
            except _UnhashableError as e:
                raise UnhashableValueError(
                    self._field_name, e.value, doc_id, e.path
                )
            self._value_to_doc_ids[hashable_value].add(doc_id)
            self._doc_id_to_value[doc_id] = hashable_value

    def remove(self, doc_id: int) -> None:
        """
        Remove a document from the index.

        :param doc_id: The document ID to remove
        """
        if doc_id in self._doc_id_to_value:
            old_value = self._doc_id_to_value[doc_id]
            self._value_to_doc_ids[old_value].discard(doc_id)
            # Clean up empty sets
            if not self._value_to_doc_ids[old_value]:
                del self._value_to_doc_ids[old_value]
            del self._doc_id_to_value[doc_id]

    def update(self, doc_id: int, document: Dict) -> None:
        """
        Update a document in the index.

        :param doc_id: The document ID
        :param document: The updated document data
        """
        # Remove old entry and add new one
        self.remove(doc_id)
        self.add(doc_id, document)

    def clear(self) -> None:
        """Clear all entries from the index."""
        self._value_to_doc_ids.clear()
        self._doc_id_to_value.clear()

    def get_eq(self, value: Any) -> Set[int]:
        """
        Get document IDs where the field equals the given value.

        :param value: The value to search for
        :returns: Set of document IDs with matching value
        """
        hashable_value = self._make_hashable(value)
        return self._value_to_doc_ids.get(hashable_value, set()).copy()

    def get_ne(self, value: Any) -> Set[int]:
        """
        Get document IDs where the field does not equal the given value.

        :param value: The value to exclude
        :returns: Set of document IDs with non-matching value
        """
        hashable_value = self._make_hashable(value)
        result = set()
        for v, doc_ids in self._value_to_doc_ids.items():
            if v != hashable_value:
                result.update(doc_ids)
        return result

    def get_lt(self, value: Any) -> Set[int]:
        """
        Get document IDs where the field is less than the given value.

        :param value: The upper bound (exclusive)
        :returns: Set of document IDs with values less than the bound
        """
        result = set()
        for v, doc_ids in self._value_to_doc_ids.items():
            try:
                if v is not None and v < value:
                    result.update(doc_ids)
            except TypeError:
                # Skip values that can't be compared
                pass
        return result

    def get_le(self, value: Any) -> Set[int]:
        """
        Get document IDs where the field is less than or equal to the value.

        :param value: The upper bound (inclusive)
        :returns: Set of document IDs with values <= the bound
        """
        result = set()
        for v, doc_ids in self._value_to_doc_ids.items():
            try:
                if v is not None and v <= value:
                    result.update(doc_ids)
            except TypeError:
                pass
        return result

    def get_gt(self, value: Any) -> Set[int]:
        """
        Get document IDs where the field is greater than the given value.

        :param value: The lower bound (exclusive)
        :returns: Set of document IDs with values greater than the bound
        """
        result = set()
        for v, doc_ids in self._value_to_doc_ids.items():
            try:
                if v is not None and v > value:
                    result.update(doc_ids)
            except TypeError:
                pass
        return result

    def get_ge(self, value: Any) -> Set[int]:
        """
        Get document IDs where the field is greater than or equal to the value.

        :param value: The lower bound (inclusive)
        :returns: Set of document IDs with values >= the bound
        """
        result = set()
        for v, doc_ids in self._value_to_doc_ids.items():
            try:
                if v is not None and v >= value:
                    result.update(doc_ids)
            except TypeError:
                pass
        return result

    def get_one_of(self, values: List[Any]) -> Set[int]:
        """
        Get document IDs where the field value is in the given list.

        :param values: List of values to match
        :returns: Set of document IDs with matching values
        """
        result = set()
        for value in values:
            result.update(self.get_eq(value))
        return result

    def get_all_doc_ids(self) -> Set[int]:
        """
        Get all document IDs in this index.

        :returns: Set of all indexed document IDs
        """
        return set(self._doc_id_to_value.keys())

    def __len__(self) -> int:
        """Return the number of documents in this index."""
        return len(self._doc_id_to_value)

    def __repr__(self) -> str:
        return f"Index(field='{self._field_name}', docs={len(self)})"


class IndexManager:
    """
    Manages indexes for a table.

    The IndexManager is responsible for:
    - Creating and dropping indexes
    - Keeping indexes in sync with document changes
    - Determining when a query can use an index
    - Executing indexed queries
    """

    def __init__(self):
        """Create a new IndexManager."""
        self._indexes: Dict[str, Index] = {}

    def create_index(self, field_name: str, documents: Dict[int, Dict]) -> None:
        """
        Create a new index on a field.

        :param field_name: The field to index (supports dot notation for nested)
        :param documents: Current documents to populate the index with
        :raises ValueError: If an index already exists for this field
        """
        if field_name in self._indexes:
            raise ValueError(f"Index already exists for field '{field_name}'")

        index = Index(field_name)
        # Populate the index with existing documents
        for doc_id, doc in documents.items():
            index.add(doc_id, doc)

        self._indexes[field_name] = index

    def drop_index(self, field_name: str) -> None:
        """
        Remove an index.

        :param field_name: The field whose index to remove
        :raises ValueError: If no index exists for this field
        """
        if field_name not in self._indexes:
            raise ValueError(f"No index exists for field '{field_name}'")

        del self._indexes[field_name]

    def has_index(self, field_name: str) -> bool:
        """
        Check if an index exists for a field.

        :param field_name: The field to check
        :returns: True if an index exists
        """
        return field_name in self._indexes

    def get_index(self, field_name: str) -> Optional[Index]:
        """
        Get an index by field name.

        :param field_name: The field name
        :returns: The Index or None if not found
        """
        return self._indexes.get(field_name)

    def list_indexes(self) -> List[str]:
        """
        List all indexed field names.

        :returns: List of field names that have indexes
        """
        return list(self._indexes.keys())

    def clear_all(self) -> None:
        """Clear all indexes."""
        self._indexes.clear()

    def on_insert(self, doc_id: int, document: Dict) -> None:
        """
        Update indexes when a document is inserted.

        :param doc_id: The new document's ID
        :param document: The document data
        """
        for index in self._indexes.values():
            index.add(doc_id, document)

    def on_update(self, doc_id: int, document: Dict) -> None:
        """
        Update indexes when a document is updated.

        :param doc_id: The document's ID
        :param document: The updated document data
        """
        for index in self._indexes.values():
            index.update(doc_id, document)

    def on_remove(self, doc_id: int) -> None:
        """
        Update indexes when a document is removed.

        :param doc_id: The removed document's ID
        """
        for index in self._indexes.values():
            index.remove(doc_id)

    def rebuild_all(self, documents: Dict[int, Dict]) -> None:
        """
        Rebuild all indexes from scratch.

        :param documents: All documents in the table
        """
        for index in self._indexes.values():
            index.clear()
            for doc_id, doc in documents.items():
                index.add(doc_id, doc)

    def try_execute_query(
        self,
        query_hash: Optional[Tuple],
        all_doc_ids: Set[int]
    ) -> Optional[Set[int]]:
        """
        Try to execute a query using indexes.

        Analyzes the query hash to determine if it can be satisfied using
        indexes. Returns the matching document IDs if successful, or None
        if the query cannot use indexes.

        :param query_hash: The query's hash tuple from QueryInstance
        :param all_doc_ids: Set of all document IDs (for fallback operations)
        :returns: Set of matching document IDs, or None if index can't be used
        """
        if query_hash is None:
            return None

        return self._execute_indexed(query_hash, all_doc_ids)

    def _execute_indexed(
        self,
        query_hash: Tuple,
        all_doc_ids: Set[int]
    ) -> Optional[Set[int]]:
        """
        Internal method to execute an indexed query.

        :param query_hash: The query hash tuple
        :param all_doc_ids: Set of all document IDs
        :returns: Set of matching doc IDs or None if cannot use index
        """
        if not query_hash:
            return None

        op = query_hash[0]

        # Handle simple equality/comparison queries
        if op in ('==', '!=', '<', '<=', '>', '>='):
            return self._execute_comparison(query_hash)

        # Handle path-based queries with exists
        if op == 'exists':
            return self._execute_exists(query_hash)

        # Handle one_of queries
        if op == 'one_of':
            return self._execute_one_of(query_hash)

        # Handle compound AND queries
        if op == 'and':
            return self._execute_and(query_hash, all_doc_ids)

        # Handle compound OR queries
        if op == 'or':
            return self._execute_or(query_hash, all_doc_ids)

        # Handle NOT queries
        if op == 'not':
            return self._execute_not(query_hash, all_doc_ids)

        return None

    def _execute_comparison(self, query_hash: Tuple) -> Optional[Set[int]]:
        """Execute a comparison query using an index."""
        if len(query_hash) < 3:
            return None

        op, path, value = query_hash[0], query_hash[1], query_hash[2]

        # path should be a tuple of field names
        if not isinstance(path, tuple) or not path:
            return None

        # Convert path tuple to dot notation
        field_name = '.'.join(str(p) for p in path)

        index = self._indexes.get(field_name)
        if index is None:
            return None

        # Execute the appropriate index operation
        if op == '==':
            return index.get_eq(value)
        elif op == '!=':
            return index.get_ne(value)
        elif op == '<':
            return index.get_lt(value)
        elif op == '<=':
            return index.get_le(value)
        elif op == '>':
            return index.get_gt(value)
        elif op == '>=':
            return index.get_ge(value)

        return None

    def _execute_exists(self, query_hash: Tuple) -> Optional[Set[int]]:
        """Execute an exists query using an index."""
        if len(query_hash) < 2:
            return None

        path = query_hash[1]
        if not isinstance(path, tuple) or not path:
            return None

        field_name = '.'.join(str(p) for p in path)
        index = self._indexes.get(field_name)
        if index is None:
            return None

        return index.get_all_doc_ids()

    def _execute_one_of(self, query_hash: Tuple) -> Optional[Set[int]]:
        """Execute a one_of query using an index."""
        if len(query_hash) < 3:
            return None

        path, values = query_hash[1], query_hash[2]
        if not isinstance(path, tuple) or not path:
            return None

        field_name = '.'.join(str(p) for p in path)
        index = self._indexes.get(field_name)
        if index is None:
            return None

        # Convert frozen values back to list
        if isinstance(values, (frozenset, tuple)):
            values = list(values)
        elif not isinstance(values, list):
            return None

        return index.get_one_of(values)

    def _execute_and(
        self,
        query_hash: Tuple,
        all_doc_ids: Set[int]
    ) -> Optional[Set[int]]:
        """Execute a compound AND query."""
        if len(query_hash) < 2:
            return None

        # query_hash[1] is a frozenset of sub-query hashes
        sub_queries = query_hash[1]
        if not isinstance(sub_queries, frozenset):
            return None

        results = []
        for sub_hash in sub_queries:
            sub_result = self._execute_indexed(sub_hash, all_doc_ids)
            if sub_result is None:
                # One part can't use index, so the whole AND can't
                return None
            results.append(sub_result)

        if not results:
            return None

        # Intersect all results
        final = results[0]
        for r in results[1:]:
            final = final & r

        return final

    def _execute_or(
        self,
        query_hash: Tuple,
        all_doc_ids: Set[int]
    ) -> Optional[Set[int]]:
        """Execute a compound OR query."""
        if len(query_hash) < 2:
            return None

        sub_queries = query_hash[1]
        if not isinstance(sub_queries, frozenset):
            return None

        results = []
        for sub_hash in sub_queries:
            sub_result = self._execute_indexed(sub_hash, all_doc_ids)
            if sub_result is None:
                # One part can't use index, so the whole OR can't
                return None
            results.append(sub_result)

        if not results:
            return None

        # Union all results
        final = set()
        for r in results:
            final = final | r

        return final

    def _execute_not(
        self,
        query_hash: Tuple,
        all_doc_ids: Set[int]
    ) -> Optional[Set[int]]:
        """Execute a NOT query."""
        if len(query_hash) < 2:
            return None

        inner_hash = query_hash[1]
        inner_result = self._execute_indexed(inner_hash, all_doc_ids)
        if inner_result is None:
            return None

        # NOT means all docs except those matching
        return all_doc_ids - inner_result

    def __repr__(self) -> str:
        index_names = list(self._indexes.keys())
        return f"IndexManager(indexes={index_names})"
