"""
This module implements tables, the central place for accessing and manipulating
data in TinyDB.
"""

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Union,
    cast,
    Tuple
)

from .queries import QueryLike
from .storages import Storage
from .utils import LRUCache
from .hooks import HookEvent, HookManager

__all__ = ('Document', 'Table')

#: The key used to mark documents as soft-deleted
SOFT_DELETE_KEY = '_deleted'


class Document(dict):
    """
    A document stored in the database.

    This class provides a way to access both a document's content and
    its ID using ``doc.doc_id``.
    """

    def __init__(self, value: Mapping, doc_id: int):
        super().__init__(value)
        self.doc_id = doc_id


class Table:
    """
    Represents a single TinyDB table.

    It provides methods for accessing and manipulating documents.

    .. admonition:: Query Cache

        As an optimization, a query cache is implemented using a
        :class:`~tinydb.utils.LRUCache`. This class mimics the interface of
        a normal ``dict``, but starts to remove the least-recently used entries
        once a threshold is reached.

        The query cache is updated on every search operation. When writing
        data, the whole cache is discarded as the query results may have
        changed.

    .. admonition:: Customization

        For customization, the following class variables can be set:

        - ``document_class`` defines the class that is used to represent
          documents,
        - ``document_id_class`` defines the class that is used to represent
          document IDs,
        - ``query_cache_class`` defines the class that is used for the query
          cache
        - ``default_query_cache_capacity`` defines the default capacity of
          the query cache

        .. versionadded:: 4.0


    :param storage: The storage instance to use for this table
    :param name: The table name
    :param cache_size: Maximum capacity of query cache
    :param persist_empty: Store new table even with no operations on it
    """

    #: The class used to represent documents
    #:
    #: .. versionadded:: 4.0
    document_class = Document

    #: The class used to represent a document ID
    #:
    #: .. versionadded:: 4.0
    document_id_class = int

    #: The class used for caching query results
    #:
    #: .. versionadded:: 4.0
    query_cache_class = LRUCache

    #: The default capacity of the query cache
    #:
    #: .. versionadded:: 4.0
    default_query_cache_capacity = 10

    def __init__(
        self,
        storage: Storage,
        name: str,
        cache_size: int = default_query_cache_capacity,
        persist_empty: bool = False,
        hooks: Optional[HookManager] = None
    ):
        """
        Create a table instance.
        """

        self._storage = storage
        self._name = name
        self._query_cache: LRUCache[QueryLike, List[Document]] \
            = self.query_cache_class(capacity=cache_size)
        self._hooks = hooks

        self._next_id = None
        if persist_empty:
            self._update_table(lambda table: table.clear())

    def __repr__(self):
        args = [
            'name={!r}'.format(self.name),
            'total={}'.format(len(self)),
            'storage={}'.format(self._storage),
        ]

        return '<{} {}>'.format(type(self).__name__, ', '.join(args))

    @property
    def name(self) -> str:
        """
        Get the table name.
        """
        return self._name

    @property
    def storage(self) -> Storage:
        """
        Get the table storage instance.
        """
        return self._storage

    def insert(self, document: Mapping) -> int:
        """
        Insert a new document into the table.

        :param document: the document to insert
        :returns: the inserted document's ID
        """

        # Make sure the document implements the ``Mapping`` interface
        if not isinstance(document, Mapping):
            raise ValueError('Document is not a Mapping')

        # First, we get the document ID for the new document
        if isinstance(document, self.document_class):
            # For a `Document` object we use the specified ID
            doc_id = document.doc_id

            # We also reset the stored next ID so the next insert won't
            # re-use document IDs by accident when storing an old value
            self._next_id = None
        else:
            # In all other cases we use the next free ID
            doc_id = self._get_next_id()

        # Prepare a copy of the document for hooks
        doc_copy = dict(document)

        # Run before-insert hooks
        self._run_hooks(
            HookEvent.BEFORE_INSERT,
            [{'doc_id': doc_id, **doc_copy}]
        )

        # Now, we update the table and add the document
        def updater(table: dict):
            if doc_id in table:
                raise ValueError(f'Document with ID {str(doc_id)} '
                                 f'already exists')

            # By calling ``dict(document)`` we convert the data we got to a
            # ``dict`` instance even if it was a different class that
            # implemented the ``Mapping`` interface
            table[doc_id] = dict(document)

        # See below for details on ``Table._update``
        self._update_table(updater)

        # Run after-insert hooks
        self._run_hooks(
            HookEvent.AFTER_INSERT,
            [{'doc_id': doc_id, **doc_copy}]
        )

        return doc_id

    def insert_multiple(self, documents: Iterable[Mapping]) -> List[int]:
        """
        Insert multiple documents into the table.

        :param documents: an Iterable of documents to insert
        :returns: a list containing the inserted documents' IDs
        """
        doc_ids: List[int] = []
        # Store document copies for hooks (built during updater)
        hook_docs: List[Dict] = []

        def updater(table: dict):
            for document in documents:

                # Make sure the document implements the ``Mapping`` interface
                if not isinstance(document, Mapping):
                    raise ValueError('Document is not a Mapping')

                if isinstance(document, self.document_class):
                    # Check if document does not override an existing document
                    if document.doc_id in table:
                        raise ValueError(
                            f'Document with ID {str(document.doc_id)} '
                            f'already exists'
                        )

                    # Store the doc_id, so we can return all document IDs
                    # later. Then save the document with its doc_id and
                    # skip the rest of the current loop
                    doc_id = document.doc_id
                    doc_ids.append(doc_id)
                    doc_copy = dict(document)
                    hook_docs.append({'doc_id': doc_id, **doc_copy})
                    table[doc_id] = doc_copy
                    continue

                # Generate new document ID for this document
                # Store the doc_id, so we can return all document IDs
                # later, then save the document with the new doc_id
                doc_id = self._get_next_id()
                doc_ids.append(doc_id)
                doc_copy = dict(document)
                hook_docs.append({'doc_id': doc_id, **doc_copy})
                table[doc_id] = doc_copy

        # Run before-insert hooks (we need to pre-process documents first)
        # Convert documents to list to allow iteration twice
        documents = list(documents)

        # Pre-calculate doc_ids for before hooks
        before_hook_docs: List[Dict] = []
        temp_next_id = self._next_id
        table_snapshot = self._read_table(include_deleted=True)
        for document in documents:
            if isinstance(document, self.document_class):
                doc_id = document.doc_id
            else:
                if temp_next_id is not None:
                    doc_id = temp_next_id
                    temp_next_id = doc_id + 1
                elif not table_snapshot:
                    doc_id = 1
                    temp_next_id = 2
                else:
                    max_id = max(self.document_id_class(i) for i in table_snapshot.keys())
                    doc_id = max_id + 1
                    temp_next_id = doc_id + 1
            before_hook_docs.append({'doc_id': doc_id, **dict(document)})

        self._run_hooks(HookEvent.BEFORE_INSERT, before_hook_docs)

        # See below for details on ``Table._update``
        self._update_table(updater)

        # Run after-insert hooks
        self._run_hooks(HookEvent.AFTER_INSERT, hook_docs)

        return doc_ids

    def all(
        self,
        limit: Optional[int] = None,
        skip: int = 0,
        include_deleted: bool = False
    ) -> List[Document]:
        """
        Get all documents stored in the table.

        :param limit: maximum number of documents to return (default: no limit)
        :param skip: number of documents to skip from the beginning
                     (default: 0)
        :param include_deleted: If True, include soft-deleted documents.
                               Default is False.
        :returns: a list with all documents.

        Example usage:

        >>> # Get first 10 documents
        >>> db.all(limit=10)
        >>> # Skip first 20 and get next 10 documents
        >>> db.all(limit=10, skip=20)
        >>> # Skip first 5 documents
        >>> db.all(skip=5)
        """

        # Get documents from the table, optionally including deleted ones
        docs = [
            self.document_class(doc, self.document_id_class(doc_id))
            for doc_id, doc in self._read_table(
                include_deleted=include_deleted
            ).items()
        ]
        return self._apply_pagination(docs, skip, limit)

    def search(
        self,
        cond: QueryLike,
        limit: Optional[int] = None,
        skip: int = 0,
        include_deleted: bool = False
    ) -> List[Document]:
        """
        Search for all documents matching a 'where' cond.

        :param cond: the condition to check against
        :param limit: maximum number of documents to return (default: no limit)
        :param skip: number of documents to skip from the beginning
                     (default: 0)
        :param include_deleted: If True, include soft-deleted documents.
                               Default is False.
        :returns: list of matching documents

        Example usage:

        >>> # Get first 10 matching documents
        >>> db.search(where('type') == 'user', limit=10)
        >>> # Skip first 20 and get next 10 matching documents
        >>> db.search(where('type') == 'user', limit=10, skip=20)
        >>> # Skip first 5 matching documents
        >>> db.search(where('type') == 'user', skip=5)
        """

        # Note: We don't use the query cache when include_deleted is True
        # because the cache only stores results for non-deleted documents
        if not include_deleted:
            # First, we check the query cache to see if it has results for
            # this query
            cached_results = self._query_cache.get(cond)
            if cached_results is not None:
                # Apply skip and limit to cached results
                docs = cached_results[:]
                return self._apply_pagination(docs, skip, limit)

        # Perform the search by applying the query to all documents.
        # Then, only if the document matches the query, convert it
        # to the document class and document ID class.
        docs = [
            self.document_class(doc, self.document_id_class(doc_id))
            for doc_id, doc in self._read_table(
                include_deleted=include_deleted
            ).items()
            if cond(doc)
        ]

        # Only cache cacheable queries when not including deleted documents
        if not include_deleted:
            # Only cache cacheable queries.
            #
            # This weird `getattr` dance is needed to make MyPy happy as
            # it doesn't know that a query might have a `is_cacheable` method
            # that is not declared in the `QueryLike` protocol due to it being
            # optional.
            # See: https://github.com/python/mypy/issues/1424
            #
            # Note also that by default we expect custom query objects to be
            # cacheable (which means they need to have a stable hash value).
            # This is to keep consistency with TinyDB's behavior before
            # `is_cacheable` was introduced which assumed that all queries
            # are cacheable.
            is_cacheable: Callable[[], bool] = getattr(cond, 'is_cacheable',
                                                       lambda: True)
            if is_cacheable():
                # Update the query cache with full results (before pagination)
                self._query_cache[cond] = docs[:]

        # Apply skip and limit to results
        return self._apply_pagination(docs, skip, limit)

    def get(
        self,
        cond: Optional[QueryLike] = None,
        doc_id: Optional[int] = None,
        doc_ids: Optional[List] = None,
        include_deleted: bool = False
    ) -> Optional[Union[Document, List[Document]]]:
        """
        Get exactly one document specified by a query or a document ID.
        However, if multiple document IDs are given then returns all
        documents in a list.

        Returns ``None`` if the document doesn't exist.

        :param cond: the condition to check against
        :param doc_id: the document's ID
        :param doc_ids: the document's IDs(multiple)
        :param include_deleted: If True, include soft-deleted documents.
                               Default is False.

        :returns: the document(s) or ``None``
        """
        table = self._read_table(include_deleted=include_deleted)

        if doc_id is not None:
            # Retrieve a document specified by its ID
            raw_doc = table.get(str(doc_id), None)

            if raw_doc is None:
                return None

            # Convert the raw data to the document class
            return self.document_class(raw_doc, doc_id)

        elif doc_ids is not None:
            # Filter the table by extracting out all those documents which
            # have doc id specified in the doc_id list.

            # Since document IDs will be unique, we make it a set to ensure
            # constant time lookup
            doc_ids_set = set(str(doc_id) for doc_id in doc_ids)

            # Now return the filtered documents in form of list
            return [
                self.document_class(doc, self.document_id_class(doc_id))
                for doc_id, doc in table.items()
                if doc_id in doc_ids_set
            ]

        elif cond is not None:
            # Find a document specified by a query
            # The trailing underscore in doc_id_ is needed so MyPy
            # doesn't think that `doc_id_` (which is a string) needs
            # to have the same type as `doc_id` which is this function's
            # parameter and is an optional `int`.
            for doc_id_, doc in self._read_table(
                include_deleted=include_deleted
            ).items():
                if cond(doc):
                    return self.document_class(
                        doc,
                        self.document_id_class(doc_id_)
                    )

            return None

        raise RuntimeError('You have to pass either cond or doc_id or doc_ids')

    def contains(
        self,
        cond: Optional[QueryLike] = None,
        doc_id: Optional[int] = None,
        include_deleted: bool = False
    ) -> bool:
        """
        Check whether the database contains a document matching a query or
        an ID.

        If ``doc_id`` is set, it checks if the db contains the specified ID.

        :param cond: the condition use
        :param doc_id: the document ID to look for
        :param include_deleted: If True, include soft-deleted documents.
                               Default is False.
        """
        if doc_id is not None:
            # Documents specified by ID
            return self.get(doc_id=doc_id, include_deleted=include_deleted) \
                is not None

        elif cond is not None:
            # Document specified by condition
            return self.get(cond, include_deleted=include_deleted) is not None

        raise RuntimeError('You have to pass either cond or doc_id')

    def update(
        self,
        fields: Union[Mapping, Callable[[Mapping], None]],
        cond: Optional[QueryLike] = None,
        doc_ids: Optional[Iterable[int]] = None,
    ) -> List[int]:
        """
        Update all matching documents to have a given set of fields.

        :param fields: the fields that the matching documents will have
                       or a method that will update the documents
        :param cond: which documents to update
        :param doc_ids: a list of document IDs
        :returns: a list containing the updated document's ID
        """

        # Check if hooks are registered to avoid unnecessary work
        has_hooks = self._has_hooks(HookEvent.BEFORE_UPDATE, HookEvent.AFTER_UPDATE)

        # Store documents for hooks (only if hooks are registered)
        before_hook_docs: List[Dict[str, Any]] = []
        after_hook_docs: List[Dict[str, Any]] = []

        # Define the function that will perform the update
        if callable(fields):
            def perform_update(table, doc_id):
                # Update documents by calling the update function provided by
                # the user
                fields(table[doc_id])
        else:
            def perform_update(table, doc_id):
                # Update documents by setting all fields from the provided data
                table[doc_id].update(fields)

        if doc_ids is not None:
            # Perform the update operation for documents specified by a list
            # of document IDs

            updated_ids = list(doc_ids)

            # Prepare before-hook documents (only if hooks are registered)
            if has_hooks:
                table_snapshot = self._read_table(include_deleted=True)
                for doc_id in updated_ids:
                    str_doc_id = str(doc_id)
                    if str_doc_id in table_snapshot:
                        before_hook_docs.append({
                            'doc_id': doc_id,
                            **dict(table_snapshot[str_doc_id])
                        })

                # Run before-update hooks
                self._run_hooks(HookEvent.BEFORE_UPDATE, before_hook_docs)

            def updater(table: dict):
                # Call the processing callback with all document IDs
                for doc_id in updated_ids:
                    perform_update(table, doc_id)
                    # Capture updated document for after hooks (only if needed)
                    if has_hooks:
                        after_hook_docs.append({
                            'doc_id': doc_id,
                            **dict(table[doc_id])
                        })

            # Perform the update operation (see _update_table for details)
            self._update_table(updater)

            # Run after-update hooks
            if has_hooks:
                self._run_hooks(HookEvent.AFTER_UPDATE, after_hook_docs)

            return updated_ids

        elif cond is not None:
            # Perform the update operation for documents specified by a query

            # Collect affected doc_ids
            updated_ids: List[int] = []

            def updater(table: dict):
                _cond = cast(QueryLike, cond)

                # We need to convert the keys iterator to a list because
                # we may remove entries from the ``table`` dict during
                # iteration and doing this without the list conversion would
                # result in an exception (RuntimeError: dictionary changed size
                # during iteration)
                for doc_id in list(table.keys()):
                    # Pass through all documents to find documents matching the
                    # query. Call the processing callback with the document ID
                    if _cond(table[doc_id]):
                        # Capture document before update for hooks (only if needed)
                        if has_hooks:
                            before_hook_docs.append({
                                'doc_id': doc_id,
                                **dict(table[doc_id])
                            })

                        # Add ID to list of updated documents
                        updated_ids.append(doc_id)

                        # Perform the update (see above)
                        perform_update(table, doc_id)

                        # Capture updated document for after hooks (only if needed)
                        if has_hooks:
                            after_hook_docs.append({
                                'doc_id': doc_id,
                                **dict(table[doc_id])
                            })

            # Run before-update hooks (need to find matching docs first)
            if has_hooks:
                table_snapshot = self._read_table(include_deleted=True)
                _cond = cast(QueryLike, cond)
                for str_doc_id, doc in table_snapshot.items():
                    if _cond(doc):
                        before_hook_docs.append({
                            'doc_id': self.document_id_class(str_doc_id),
                            **dict(doc)
                        })
                self._run_hooks(HookEvent.BEFORE_UPDATE, before_hook_docs)

                # Reset for actual update
                before_hook_docs.clear()

            # Perform the update operation (see _update_table for details)
            self._update_table(updater)

            # Run after-update hooks
            if has_hooks:
                self._run_hooks(HookEvent.AFTER_UPDATE, after_hook_docs)

            return updated_ids

        else:
            # Update all documents unconditionally

            updated_ids = []

            # Prepare before-hook documents (only if hooks are registered)
            if has_hooks:
                table_snapshot = self._read_table(include_deleted=True)
                for str_doc_id, doc in table_snapshot.items():
                    before_hook_docs.append({
                        'doc_id': self.document_id_class(str_doc_id),
                        **dict(doc)
                    })

                # Run before-update hooks
                self._run_hooks(HookEvent.BEFORE_UPDATE, before_hook_docs)

            def updater(table: dict):
                # Process all documents
                for doc_id in list(table.keys()):
                    # Add ID to list of updated documents
                    updated_ids.append(doc_id)

                    # Perform the update (see above)
                    perform_update(table, doc_id)

                    # Capture updated document for after hooks (only if needed)
                    if has_hooks:
                        after_hook_docs.append({
                            'doc_id': doc_id,
                            **dict(table[doc_id])
                        })

            # Perform the update operation (see _update_table for details)
            self._update_table(updater)

            # Run after-update hooks
            if has_hooks:
                self._run_hooks(HookEvent.AFTER_UPDATE, after_hook_docs)

            return updated_ids

    def update_multiple(
        self,
        updates: Iterable[
            Tuple[Union[Mapping, Callable[[Mapping], None]], QueryLike]
        ],
    ) -> List[int]:
        """
        Update all matching documents to have a given set of fields.

        :returns: a list containing the updated document's ID
        """

        # Check if hooks are registered to avoid unnecessary work
        has_hooks = self._has_hooks(HookEvent.BEFORE_UPDATE, HookEvent.AFTER_UPDATE)

        # Store documents for hooks (only if hooks are registered)
        before_hook_docs: List[Dict[str, Any]] = []
        after_hook_docs: List[Dict[str, Any]] = []

        # Convert updates to list to allow multiple iterations
        updates = list(updates)

        # Define the function that will perform the update
        def perform_update(fields, table, doc_id):
            if callable(fields):
                # Update documents by calling the update function provided
                # by the user
                fields(table[doc_id])
            else:
                # Update documents by setting all fields from the provided
                # data
                table[doc_id].update(fields)

        # Pre-calculate which documents will be affected for before hooks
        # (only if hooks are registered)
        if has_hooks:
            table_snapshot = self._read_table(include_deleted=True)
            affected_doc_ids = set()
            for str_doc_id, doc in table_snapshot.items():
                for fields, cond in updates:
                    _cond = cast(QueryLike, cond)
                    if _cond(doc):
                        doc_id = self.document_id_class(str_doc_id)
                        if doc_id not in affected_doc_ids:
                            affected_doc_ids.add(doc_id)
                            before_hook_docs.append({
                                'doc_id': doc_id,
                                **dict(doc)
                            })

            # Run before-update hooks
            self._run_hooks(HookEvent.BEFORE_UPDATE, before_hook_docs)

        # Perform the update operation for documents specified by a query

        # Collect affected doc_ids
        updated_ids: List[int] = []

        def updater(table: dict):
            # We need to convert the keys iterator to a list because
            # we may remove entries from the ``table`` dict during
            # iteration and doing this without the list conversion would
            # result in an exception (RuntimeError: dictionary changed size
            # during iteration)
            for doc_id in list(table.keys()):
                for fields, cond in updates:
                    _cond = cast(QueryLike, cond)

                    # Pass through all documents to find documents matching the
                    # query. Call the processing callback with the document ID
                    if _cond(table[doc_id]):
                        # Add ID to list of updated documents
                        updated_ids.append(doc_id)

                        # Perform the update (see above)
                        perform_update(fields, table, doc_id)

                        # Capture updated document for after hooks (only if needed)
                        if has_hooks:
                            after_hook_docs.append({
                                'doc_id': doc_id,
                                **dict(table[doc_id])
                            })

        # Perform the update operation (see _update_table for details)
        self._update_table(updater)

        # Run after-update hooks
        if has_hooks:
            self._run_hooks(HookEvent.AFTER_UPDATE, after_hook_docs)

        return updated_ids

    def upsert(self, document: Mapping, cond: Optional[QueryLike] = None) -> List[int]:
        """
        Update documents, if they exist, insert them otherwise.

        Note: This will update *all* documents matching the query. Document
        argument can be a tinydb.table.Document object if you want to specify a
        doc_id.

        :param document: the document to insert or the fields to update
        :param cond: which document to look for, optional if you've passed a
        Document with a doc_id
        :returns: a list containing the updated documents' IDs
        """

        # Extract doc_id
        if isinstance(document, self.document_class) and hasattr(document, 'doc_id'):
            doc_ids: Optional[List[int]] = [document.doc_id]
        else:
            doc_ids = None

        # Make sure we can actually find a matching document
        if doc_ids is None and cond is None:
            raise ValueError("If you don't specify a search query, you must "
                             "specify a doc_id. Hint: use a table.Document "
                             "object.")

        # Perform the update operation
        try:
            updated_docs: Optional[List[int]] = self.update(document, cond, doc_ids)
        except KeyError:
            # This happens when a doc_id is specified, but it's missing
            updated_docs = None

        # If documents have been updated: return their IDs
        if updated_docs:
            return updated_docs

        # There are no documents that match the specified query -> insert the
        # data as a new document
        return [self.insert(document)]

    def remove(
        self,
        cond: Optional[QueryLike] = None,
        doc_ids: Optional[Iterable[int]] = None,
    ) -> List[int]:
        """
        Remove all matching documents.

        :param cond: the condition to check against
        :param doc_ids: a list of document IDs
        :returns: a list containing the removed documents' ID
        """
        # Store documents for hooks
        hook_docs: List[Dict[str, Any]] = []

        if doc_ids is not None:
            # This function returns the list of IDs for the documents that have
            # been removed. When removing documents identified by a set of
            # document IDs, it's this list of document IDs we need to return
            # later.
            # We convert the document ID iterator into a list, so we can both
            # use the document IDs to remove the specified documents and
            # to return the list of affected document IDs
            removed_ids = list(doc_ids)

            # Prepare documents for hooks
            table_snapshot = self._read_table(include_deleted=True)
            for doc_id in removed_ids:
                str_doc_id = str(doc_id)
                if str_doc_id in table_snapshot:
                    hook_docs.append({
                        'doc_id': doc_id,
                        **dict(table_snapshot[str_doc_id])
                    })

            # Run before-delete hooks
            self._run_hooks(HookEvent.BEFORE_DELETE, hook_docs)

            def updater(table: dict):
                for doc_id in removed_ids:
                    table.pop(doc_id)

            # Perform the remove operation
            self._update_table(updater)

            # Run after-delete hooks
            self._run_hooks(HookEvent.AFTER_DELETE, hook_docs)

            return removed_ids

        if cond is not None:
            removed_ids: List[int] = []

            # Pre-calculate which documents will be removed for before hooks
            table_snapshot = self._read_table(include_deleted=True)
            _cond = cast(QueryLike, cond)
            for str_doc_id, doc in table_snapshot.items():
                if _cond(doc):
                    hook_docs.append({
                        'doc_id': self.document_id_class(str_doc_id),
                        **dict(doc)
                    })

            # Run before-delete hooks
            self._run_hooks(HookEvent.BEFORE_DELETE, hook_docs)

            # This updater function will be called with the table data
            # as its first argument. See ``Table._update`` for details on this
            # operation
            def updater(table: dict):
                # We need to convince MyPy (the static type checker) that
                # the ``cond is not None`` invariant still holds true when
                # the updater function is called
                _cond = cast(QueryLike, cond)

                # We need to convert the keys iterator to a list because
                # we may remove entries from the ``table`` dict during
                # iteration and doing this without the list conversion would
                # result in an exception (RuntimeError: dictionary changed size
                # during iteration)
                for doc_id in list(table.keys()):
                    if _cond(table[doc_id]):
                        # Add document ID to list of removed document IDs
                        removed_ids.append(doc_id)

                        # Remove document from the table
                        table.pop(doc_id)

            # Perform the remove operation
            self._update_table(updater)

            # Run after-delete hooks
            self._run_hooks(HookEvent.AFTER_DELETE, hook_docs)

            return removed_ids

        raise RuntimeError('Use truncate() to remove all documents')

    def truncate(self) -> None:
        """
        Truncate the table by removing all documents.
        """
        # Check if hooks are registered to avoid unnecessary work
        has_hooks = self._has_hooks(HookEvent.BEFORE_DELETE, HookEvent.AFTER_DELETE)

        # Prepare documents for hooks (only if hooks are registered)
        hook_docs: List[Dict[str, Any]] = []
        if has_hooks:
            table_snapshot = self._read_table(include_deleted=True)
            for str_doc_id, doc in table_snapshot.items():
                hook_docs.append({
                    'doc_id': self.document_id_class(str_doc_id),
                    **dict(doc)
                })

            # Run before-delete hooks
            self._run_hooks(HookEvent.BEFORE_DELETE, hook_docs)

        # Update the table by resetting all data
        self._update_table(lambda table: table.clear())

        # Run after-delete hooks
        if has_hooks:
            self._run_hooks(HookEvent.AFTER_DELETE, hook_docs)

        # Reset document ID counter
        self._next_id = None

    def soft_remove(
        self,
        cond: Optional[QueryLike] = None,
        doc_ids: Optional[Iterable[int]] = None,
    ) -> List[int]:
        """
        Soft delete documents by marking them as deleted instead of
        removing them permanently. Soft-deleted documents are hidden
        from normal queries but can be restored later.

        :param cond: the condition to check against
        :param doc_ids: a list of document IDs
        :returns: a list containing the soft-deleted documents' IDs

        Note: This method clears the query cache since soft-deleted
        documents are hidden from normal queries.
        """
        # Check if hooks are registered to avoid unnecessary work
        has_hooks = self._has_hooks(HookEvent.BEFORE_DELETE, HookEvent.AFTER_DELETE)

        # Store documents for hooks (only if hooks are registered)
        hook_docs: List[Dict[str, Any]] = []

        if doc_ids is not None:
            soft_deleted_ids = list(doc_ids)

            # Prepare documents for hooks (only if hooks are registered)
            if has_hooks:
                table_snapshot = self._read_table(include_deleted=True)
                for doc_id in soft_deleted_ids:
                    str_doc_id = str(doc_id)
                    if str_doc_id in table_snapshot:
                        hook_docs.append({
                            'doc_id': doc_id,
                            **dict(table_snapshot[str_doc_id])
                        })

                # Run before-delete hooks
                self._run_hooks(HookEvent.BEFORE_DELETE, hook_docs)

            def updater(table: dict):
                for doc_id in soft_deleted_ids:
                    if doc_id in table:
                        table[doc_id][SOFT_DELETE_KEY] = True

            # _update_table clears the cache after modifying data
            self._update_table(updater)

            # Run after-delete hooks
            if has_hooks:
                self._run_hooks(HookEvent.AFTER_DELETE, hook_docs)

            return soft_deleted_ids

        if cond is not None:
            soft_deleted_ids: List[int] = []

            # Pre-calculate which documents will be affected for before hooks
            # (only if hooks are registered)
            if has_hooks:
                table_snapshot = self._read_table(include_deleted=True)
                _cond = cast(QueryLike, cond)
                for str_doc_id, doc in table_snapshot.items():
                    # Only include docs that match cond and are not already deleted
                    if not doc.get(SOFT_DELETE_KEY, False) and _cond(doc):
                        hook_docs.append({
                            'doc_id': self.document_id_class(str_doc_id),
                            **dict(doc)
                        })

                # Run before-delete hooks
                self._run_hooks(HookEvent.BEFORE_DELETE, hook_docs)

            def updater(table: dict):
                _cond = cast(QueryLike, cond)

                for doc_id in list(table.keys()):
                    doc = table[doc_id]
                    # Only soft-delete if not already deleted and matches cond
                    if not doc.get(SOFT_DELETE_KEY, False) and _cond(doc):
                        soft_deleted_ids.append(doc_id)
                        table[doc_id][SOFT_DELETE_KEY] = True

            # _update_table clears the cache after modifying data
            self._update_table(updater)

            # Run after-delete hooks
            if has_hooks:
                self._run_hooks(HookEvent.AFTER_DELETE, hook_docs)

            return soft_deleted_ids

        raise RuntimeError('You have to pass either cond or doc_ids')

    def restore(
        self,
        cond: Optional[QueryLike] = None,
        doc_ids: Optional[Iterable[int]] = None,
    ) -> List[int]:
        """
        Restore soft-deleted documents, making them visible again
        in normal queries.

        When using a condition, the condition is applied to the document
        WITHOUT the internal ``_deleted`` field, so users work with
        documents as they originally stored them.

        :param cond: the condition to check against (applied to deleted docs)
        :param doc_ids: a list of document IDs to restore
        :returns: a list containing the restored documents' IDs

        Note: This method clears the query cache since restoring documents
        changes which documents are visible in normal queries.

        Note: Restore triggers INSERT hooks since documents are being
        brought back into the visible dataset.
        """
        # Check if hooks are registered to avoid unnecessary work
        has_hooks = self._has_hooks(HookEvent.BEFORE_INSERT, HookEvent.AFTER_INSERT)

        # Store documents for hooks (restore triggers INSERT hooks)
        hook_docs: List[Dict[str, Any]] = []

        if doc_ids is not None:
            restored_ids: List[int] = []
            doc_ids_list = list(doc_ids)

            # Pre-calculate which documents will be restored for before hooks
            # (only if hooks are registered)
            if has_hooks:
                table_snapshot = self._read_table(include_deleted=True)
                for doc_id in doc_ids_list:
                    str_doc_id = str(doc_id)
                    if str_doc_id in table_snapshot:
                        # Note: _read_table already strips _deleted key
                        hook_docs.append({
                            'doc_id': doc_id,
                            **dict(table_snapshot[str_doc_id])
                        })

                # Run before-insert hooks (restore brings documents back)
                self._run_hooks(HookEvent.BEFORE_INSERT, hook_docs)

            def updater(table: dict):
                for doc_id in doc_ids_list:
                    if doc_id in table and \
                            table[doc_id].get(SOFT_DELETE_KEY, False):
                        del table[doc_id][SOFT_DELETE_KEY]
                        restored_ids.append(doc_id)

            # _update_table clears the cache after modifying data
            self._update_table(updater)

            # Run after-insert hooks
            if has_hooks:
                self._run_hooks(HookEvent.AFTER_INSERT, hook_docs)

            return restored_ids

        if cond is not None:
            restored_ids = []

            # Pre-calculate which documents will be restored for before hooks
            # (only if hooks are registered)
            if has_hooks:
                # Use deleted_only to get only soft-deleted documents
                table_snapshot = self._read_table(deleted_only=True)
                _cond = cast(QueryLike, cond)
                for str_doc_id, doc in table_snapshot.items():
                    if _cond(doc):
                        hook_docs.append({
                            'doc_id': self.document_id_class(str_doc_id),
                            **dict(doc)
                        })

                # Run before-insert hooks
                self._run_hooks(HookEvent.BEFORE_INSERT, hook_docs)

            def updater(table: dict):
                _cond = cast(QueryLike, cond)

                for doc_id in list(table.keys()):
                    doc = table[doc_id]
                    # Only restore if deleted and matches condition
                    # Strip _deleted key before applying user condition
                    if doc.get(SOFT_DELETE_KEY, False):
                        clean_doc = {
                            k: v for k, v in doc.items()
                            if k != SOFT_DELETE_KEY
                        }
                        if _cond(clean_doc):
                            restored_ids.append(doc_id)
                            del table[doc_id][SOFT_DELETE_KEY]

            # _update_table clears the cache after modifying data
            self._update_table(updater)

            # Run after-insert hooks
            if has_hooks:
                self._run_hooks(HookEvent.AFTER_INSERT, hook_docs)

            return restored_ids

        # Restore all soft-deleted documents
        restored_ids = []

        # Pre-calculate all soft-deleted documents for before hooks
        # (only if hooks are registered)
        if has_hooks:
            table_snapshot = self._read_table(deleted_only=True)
            for str_doc_id, doc in table_snapshot.items():
                hook_docs.append({
                    'doc_id': self.document_id_class(str_doc_id),
                    **dict(doc)
                })

            # Run before-insert hooks
            self._run_hooks(HookEvent.BEFORE_INSERT, hook_docs)

        def updater(table: dict):
            for doc_id in list(table.keys()):
                doc = table[doc_id]
                if doc.get(SOFT_DELETE_KEY, False):
                    restored_ids.append(doc_id)
                    del table[doc_id][SOFT_DELETE_KEY]

        # _update_table clears the cache after modifying data
        self._update_table(updater)

        # Run after-insert hooks
        if has_hooks:
            self._run_hooks(HookEvent.AFTER_INSERT, hook_docs)

        return restored_ids

    def deleted(
        self,
        cond: Optional[QueryLike] = None,
        limit: Optional[int] = None,
        skip: int = 0
    ) -> List[Document]:
        """
        Get all soft-deleted documents, optionally filtered by a condition.

        The returned documents do NOT contain the internal ``_deleted`` field,
        so users can work with them as normal documents. The condition is
        also applied to documents without the ``_deleted`` field visible.

        :param cond: optional condition to filter deleted documents
        :param limit: maximum number of documents to return (default: no limit)
        :param skip: number of documents to skip from the beginning
                     (default: 0)
        :returns: a list of soft-deleted documents (without _deleted field)
        """
        # Use centralized _read_table with deleted_only=True to get
        # only soft-deleted documents (already has _deleted field stripped)
        table = self._read_table(deleted_only=True)

        # Build list of deleted documents, applying user condition if provided
        deleted_docs = [
            self.document_class(doc, self.document_id_class(doc_id))
            for doc_id, doc in table.items()
            if cond is None or cond(doc)
        ]

        return self._apply_pagination(deleted_docs, skip, limit)

    def purge(
        self,
        cond: Optional[QueryLike] = None,
        doc_ids: Optional[Iterable[int]] = None,
    ) -> List[int]:
        """
        Permanently remove soft-deleted documents from the database.
        This is a destructive operation that cannot be undone.

        If no arguments are provided, all soft-deleted documents will
        be permanently removed.

        When using a condition, the condition is applied to the document
        WITHOUT the internal ``_deleted`` field, so users work with
        documents as they originally stored them.

        :param cond: condition to filter which deleted documents to purge
        :param doc_ids: specific document IDs to purge (must be soft-deleted)
        :returns: a list containing the purged documents' IDs

        Note: This method clears the query cache since purging documents
        removes them from the database entirely.
        """
        # Check if hooks are registered to avoid unnecessary work
        has_hooks = self._has_hooks(HookEvent.BEFORE_DELETE, HookEvent.AFTER_DELETE)

        # Store documents for hooks (only if hooks are registered)
        hook_docs: List[Dict[str, Any]] = []

        if doc_ids is not None:
            purged_ids: List[int] = []
            doc_ids_list = list(doc_ids)

            # Pre-calculate which documents will be purged for before hooks
            # (only if hooks are registered)
            if has_hooks:
                # Use deleted_only to get only soft-deleted documents
                table_snapshot = self._read_table(deleted_only=True)
                for doc_id in doc_ids_list:
                    str_doc_id = str(doc_id)
                    if str_doc_id in table_snapshot:
                        hook_docs.append({
                            'doc_id': doc_id,
                            **dict(table_snapshot[str_doc_id])
                        })

                # Run before-delete hooks
                self._run_hooks(HookEvent.BEFORE_DELETE, hook_docs)

            def updater(table: dict):
                for doc_id in doc_ids_list:
                    # Only purge if the document is soft-deleted
                    if doc_id in table and \
                            table[doc_id].get(SOFT_DELETE_KEY, False):
                        table.pop(doc_id)
                        purged_ids.append(doc_id)

            # _update_table clears the cache after modifying data
            self._update_table(updater)

            # Run after-delete hooks
            if has_hooks:
                self._run_hooks(HookEvent.AFTER_DELETE, hook_docs)

            return purged_ids

        if cond is not None:
            purged_ids = []

            # Pre-calculate which documents will be purged for before hooks
            # (only if hooks are registered)
            if has_hooks:
                table_snapshot = self._read_table(deleted_only=True)
                _cond = cast(QueryLike, cond)
                for str_doc_id, doc in table_snapshot.items():
                    if _cond(doc):
                        hook_docs.append({
                            'doc_id': self.document_id_class(str_doc_id),
                            **dict(doc)
                        })

                # Run before-delete hooks
                self._run_hooks(HookEvent.BEFORE_DELETE, hook_docs)

            def updater(table: dict):
                _cond = cast(QueryLike, cond)

                for doc_id in list(table.keys()):
                    doc = table[doc_id]
                    # Only purge if soft-deleted and matches condition
                    # Strip _deleted key before applying user condition
                    if doc.get(SOFT_DELETE_KEY, False):
                        clean_doc = {
                            k: v for k, v in doc.items()
                            if k != SOFT_DELETE_KEY
                        }
                        if _cond(clean_doc):
                            purged_ids.append(doc_id)
                            table.pop(doc_id)

            # _update_table clears the cache after modifying data
            self._update_table(updater)

            # Run after-delete hooks
            if has_hooks:
                self._run_hooks(HookEvent.AFTER_DELETE, hook_docs)

            return purged_ids

        # Purge all soft-deleted documents
        purged_ids = []

        # Pre-calculate all soft-deleted documents for before hooks
        # (only if hooks are registered)
        if has_hooks:
            table_snapshot = self._read_table(deleted_only=True)
            for str_doc_id, doc in table_snapshot.items():
                hook_docs.append({
                    'doc_id': self.document_id_class(str_doc_id),
                    **dict(doc)
                })

            # Run before-delete hooks
            self._run_hooks(HookEvent.BEFORE_DELETE, hook_docs)

        def updater(table: dict):
            for doc_id in list(table.keys()):
                doc = table[doc_id]
                if doc.get(SOFT_DELETE_KEY, False):
                    purged_ids.append(doc_id)
                    table.pop(doc_id)

        # _update_table clears the cache after modifying data
        self._update_table(updater)

        # Run after-delete hooks
        if has_hooks:
            self._run_hooks(HookEvent.AFTER_DELETE, hook_docs)

        return purged_ids

    def count(self, cond: QueryLike, include_deleted: bool = False) -> int:
        """
        Count the documents matching a query.

        :param cond: the condition use
        :param include_deleted: If True, include soft-deleted documents.
                               Default is False.
        """

        return len(self.search(cond, include_deleted=include_deleted))

    def clear_cache(self) -> None:
        """
        Clear the query cache.
        """

        self._query_cache.clear()

    def _apply_pagination(
        self,
        docs: List[Document],
        skip: int,
        limit: Optional[int]
    ) -> List[Document]:
        """
        Apply skip and limit pagination to a list of documents.

        :param docs: the list of documents to paginate
        :param skip: number of documents to skip from the beginning.
                     When skip is 0 (default), no documents are skipped
                     and pagination starts from the first document.
        :param limit: maximum number of documents to return.
                      When limit is None (default), all remaining documents
                      after skip are returned (no limit applied).
        :returns: paginated list of documents

        :raises ValueError: if skip is negative or limit is negative

        Examples:
            - skip=0, limit=None: returns all documents (no pagination)
            - skip=5, limit=None: skips first 5, returns all remaining
            - skip=0, limit=10: returns first 10 documents
            - skip=5, limit=10: skips first 5, returns next 10 documents
        """
        if skip < 0:
            raise ValueError(
                f'skip must be a non-negative integer, got {skip}'
            )

        if limit is not None and limit < 0:
            raise ValueError(
                f'limit must be a non-negative integer or None, got {limit}'
            )

        if skip:
            docs = docs[skip:]

        if limit is not None:
            docs = docs[:limit]

        return docs

    def __len__(self):
        """
        Count the total number of non-deleted documents in this table.
        """

        return len(self._read_table())

    def __iter__(self) -> Iterator[Document]:
        """
        Iterate over all non-deleted documents stored in the table.

        :returns: an iterator over all non-deleted documents.
        """

        # Iterate all documents and their IDs (excluding soft-deleted)
        for doc_id, doc in self._read_table().items():
            # Convert documents to the document class
            yield self.document_class(doc, self.document_id_class(doc_id))

    def _get_next_id(self):
        """
        Return the ID for a newly inserted document.
        """

        # If we already know the next ID
        if self._next_id is not None:
            next_id = self._next_id
            self._next_id = next_id + 1

            return next_id

        # Determine the next document ID by finding out the max ID value
        # of the current table documents

        # Read the table documents
        table = self._read_table()

        # If the table is empty, set the initial ID
        if not table:
            next_id = 1
            self._next_id = next_id + 1

            return next_id

        # Determine the next ID based on the maximum ID that's currently in use
        max_id = max(self.document_id_class(i) for i in table.keys())
        next_id = max_id + 1

        # The next ID we will return AFTER this call needs to be larger than
        # the current next ID we calculated
        self._next_id = next_id + 1

        return next_id

    def _read_table(
        self,
        include_deleted: bool = False,
        deleted_only: bool = False
    ) -> Dict[str, Mapping]:
        """
        Read the table data from the underlying storage.

        Documents and doc_ids are NOT yet transformed, as
        we may not want to convert *all* documents when returning
        only one document for example.

        The returned documents always have the internal ``_deleted`` field
        stripped so users never see implementation details.

        :param include_deleted: If True, include soft-deleted documents.
                               Default is False (exclude deleted documents).
        :param deleted_only: If True, return ONLY soft-deleted documents.
                            This takes precedence over include_deleted.
        """

        # Retrieve the tables from the storage
        tables = self._storage.read()

        if tables is None:
            # The database is empty
            return {}

        # Retrieve the current table's data
        try:
            table = tables[self.name]
        except KeyError:
            # The table does not exist yet, so it is empty
            return {}

        # Apply soft-delete filtering based on parameters and always strip
        # the _deleted field so users never see internal implementation details
        if deleted_only:
            # Return only soft-deleted documents (with _deleted field stripped)
            table = {
                doc_id: self._strip_soft_delete_key(doc)
                for doc_id, doc in table.items()
                if doc.get(SOFT_DELETE_KEY, False)
            }
        elif not include_deleted:
            # Exclude soft-deleted documents (default behavior)
            # Non-deleted docs don't have _deleted field, but strip anyway
            # for consistency
            table = {
                doc_id: self._strip_soft_delete_key(doc)
                for doc_id, doc in table.items()
                if not doc.get(SOFT_DELETE_KEY, False)
            }
        else:
            # include_deleted=True: return all documents with _deleted stripped
            table = {
                doc_id: self._strip_soft_delete_key(doc)
                for doc_id, doc in table.items()
            }

        return table

    def _strip_soft_delete_key(self, doc: Mapping) -> dict:
        """
        Return a copy of the document with the soft-delete key removed.

        This ensures users don't see internal implementation details
        in their documents.

        :param doc: The document to strip the key from
        :returns: A new dict without the SOFT_DELETE_KEY
        """
        return {k: v for k, v in doc.items() if k != SOFT_DELETE_KEY}

    def _has_hooks(self, *events: HookEvent) -> bool:
        """
        Check if any hooks are registered for the given events.

        This can be used to skip expensive hook preparation (like reading
        tables and building document lists) when no hooks are registered.

        :param events: One or more hook events to check
        :returns: True if at least one hook is registered for any event
        """
        if self._hooks is None:
            return False
        return any(self._hooks.has_hooks(event) for event in events)

    def _run_hooks(
        self,
        event: HookEvent,
        documents: List[Dict]
    ) -> None:
        """
        Execute hooks for a given event if a hook manager is configured.

        This method safely runs hooks without affecting database operations.
        If no hook manager is configured or no hooks are registered for
        the event, this method does nothing.

        :param event: The hook event to trigger
        :param documents: The affected documents with their doc_ids
        """
        if self._hooks is not None and self._hooks.has_hooks(event):
            self._hooks.run(event, self._name, documents)

    def _update_table(self, updater: Callable[[Dict[int, Mapping]], None]):
        """
        Perform a table update operation.

        The storage interface used by TinyDB only allows to read/write the
        complete database data, but not modifying only portions of it. Thus,
        to only update portions of the table data, we first perform a read
        operation, perform the update on the table data and then write
        the updated data back to the storage.

        As a further optimization, we don't convert the documents into the
        document class, as the table data will *not* be returned to the user.
        """

        tables = self._storage.read()

        if tables is None:
            # The database is empty
            tables = {}

        try:
            raw_table = tables[self.name]
        except KeyError:
            # The table does not exist yet, so it is empty
            raw_table = {}

        # Convert the document IDs to the document ID class.
        # This is required as the rest of TinyDB expects the document IDs
        # to be an instance of ``self.document_id_class`` but the storage
        # might convert dict keys to strings.
        table = {
            self.document_id_class(doc_id): doc
            for doc_id, doc in raw_table.items()
        }

        # Perform the table update operation
        updater(table)

        # Convert the document IDs back to strings.
        # This is required as some storages (most notably the JSON file format)
        # don't support IDs other than strings.
        tables[self.name] = {
            str(doc_id): doc
            for doc_id, doc in table.items()
        }

        # Write the newly updated data back to the storage
        self._storage.write(tables)

        # Clear the query cache, as the table contents have changed
        self.clear_cache()
