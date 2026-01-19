"""
Tests for document timestamp metadata functionality.

This module tests the automatic tracking of document creation, update,
and deletion timestamps, as well as timestamp-based query methods.
"""
import time

import pytest

from tinydb import TinyDB, where
from tinydb.storages import MemoryStorage
from tinydb.table import (
    CREATED_AT_KEY,
    UPDATED_AT_KEY,
    DELETED_AT_KEY,
    METADATA_KEYS
)


class TestTimestampMetadataConstants:
    """Test that timestamp constants are properly defined."""

    def test_metadata_keys_defined(self):
        """Verify all metadata key constants are defined."""
        assert CREATED_AT_KEY == '_created_at'
        assert UPDATED_AT_KEY == '_updated_at'
        assert DELETED_AT_KEY == '_deleted_at'

    def test_metadata_keys_set_contains_all_keys(self):
        """Verify METADATA_KEYS contains all internal keys."""
        assert CREATED_AT_KEY in METADATA_KEYS
        assert UPDATED_AT_KEY in METADATA_KEYS
        assert DELETED_AT_KEY in METADATA_KEYS
        assert '_deleted' in METADATA_KEYS


class TestCreatedAtTimestamp:
    """Tests for automatic creation timestamp on insert."""

    def test_insert_sets_created_at(self):
        """Verify insert sets _created_at timestamp."""
        db = TinyDB(storage=MemoryStorage)
        before = time.time()
        doc_id = db.insert({'name': 'test'})
        after = time.time()

        # Get document with metadata
        doc = db.get(doc_id=doc_id, include_metadata=True)

        assert CREATED_AT_KEY in doc
        assert before <= doc[CREATED_AT_KEY] <= after

    def test_insert_multiple_sets_created_at(self):
        """Verify insert_multiple sets _created_at timestamp for all docs."""
        db = TinyDB(storage=MemoryStorage)
        before = time.time()
        doc_ids = db.insert_multiple([
            {'name': 'doc1'},
            {'name': 'doc2'},
            {'name': 'doc3'}
        ])
        after = time.time()

        # Get all documents with metadata
        docs = db.all(include_metadata=True)

        assert len(docs) == 3
        for doc in docs:
            assert CREATED_AT_KEY in doc
            assert before <= doc[CREATED_AT_KEY] <= after

    def test_created_at_not_visible_by_default(self):
        """Verify _created_at is stripped from normal queries."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'test'})

        # Get document without metadata (default)
        doc = db.get(doc_id=1)

        assert CREATED_AT_KEY not in doc
        assert doc == {'name': 'test'}

    def test_created_at_visible_with_include_metadata(self):
        """Verify _created_at is visible when include_metadata=True."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'test'})

        # Get document with metadata
        doc = db.get(doc_id=1, include_metadata=True)

        assert CREATED_AT_KEY in doc
        assert 'name' in doc


class TestUpdatedAtTimestamp:
    """Tests for automatic update timestamp on update."""

    def test_update_sets_updated_at(self):
        """Verify update sets _updated_at timestamp."""
        db = TinyDB(storage=MemoryStorage)
        doc_id = db.insert({'name': 'original'})

        # Small delay to ensure timestamps differ
        time.sleep(0.01)

        before_update = time.time()
        db.update({'name': 'updated'}, doc_ids=[doc_id])
        after_update = time.time()

        # Get document with metadata
        doc = db.get(doc_id=doc_id, include_metadata=True)

        assert UPDATED_AT_KEY in doc
        assert before_update <= doc[UPDATED_AT_KEY] <= after_update

    def test_update_with_condition_sets_updated_at(self):
        """Verify update with condition sets _updated_at timestamp."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'test', 'type': 'user'})

        time.sleep(0.01)

        before_update = time.time()
        db.update({'name': 'updated'}, where('type') == 'user')
        after_update = time.time()

        doc = db.get(where('type') == 'user', include_metadata=True)

        assert UPDATED_AT_KEY in doc
        assert before_update <= doc[UPDATED_AT_KEY] <= after_update

    def test_update_multiple_sets_updated_at(self):
        """Verify update_multiple sets _updated_at timestamp."""
        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple([
            {'name': 'doc1', 'type': 'a'},
            {'name': 'doc2', 'type': 'b'}
        ])

        time.sleep(0.01)

        before_update = time.time()
        db.update_multiple([
            ({'name': 'updated1'}, where('type') == 'a'),
            ({'name': 'updated2'}, where('type') == 'b')
        ])
        after_update = time.time()

        docs = db.all(include_metadata=True)

        for doc in docs:
            assert UPDATED_AT_KEY in doc
            assert before_update <= doc[UPDATED_AT_KEY] <= after_update

    def test_updated_at_not_visible_by_default(self):
        """Verify _updated_at is stripped from normal queries."""
        db = TinyDB(storage=MemoryStorage)
        doc_id = db.insert({'name': 'original'})
        db.update({'name': 'updated'}, doc_ids=[doc_id])

        # Get document without metadata (default)
        doc = db.get(doc_id=doc_id)

        assert UPDATED_AT_KEY not in doc
        assert doc == {'name': 'updated'}

    def test_freshly_inserted_doc_has_no_updated_at(self):
        """Verify newly inserted document has no _updated_at."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'test'})

        doc = db.get(doc_id=1, include_metadata=True)

        assert CREATED_AT_KEY in doc
        assert UPDATED_AT_KEY not in doc


class TestDeletedAtTimestamp:
    """Tests for automatic deletion timestamp on soft_remove."""

    def test_soft_remove_sets_deleted_at(self):
        """Verify soft_remove sets _deleted_at timestamp."""
        db = TinyDB(storage=MemoryStorage)
        doc_id = db.insert({'name': 'test'})

        time.sleep(0.01)

        before_delete = time.time()
        db.soft_remove(doc_ids=[doc_id])
        after_delete = time.time()

        # Get deleted document with metadata
        deleted_docs = db.deleted(include_metadata=True)

        assert len(deleted_docs) == 1
        doc = deleted_docs[0]
        assert DELETED_AT_KEY in doc
        assert before_delete <= doc[DELETED_AT_KEY] <= after_delete

    def test_soft_remove_with_condition_sets_deleted_at(self):
        """Verify soft_remove with condition sets _deleted_at timestamp."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'test', 'type': 'user'})

        time.sleep(0.01)

        before_delete = time.time()
        db.soft_remove(where('type') == 'user')
        after_delete = time.time()

        deleted_docs = db.deleted(include_metadata=True)

        assert len(deleted_docs) == 1
        doc = deleted_docs[0]
        assert DELETED_AT_KEY in doc
        assert before_delete <= doc[DELETED_AT_KEY] <= after_delete

    def test_deleted_at_not_visible_by_default(self):
        """Verify _deleted_at is stripped from deleted() by default."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'test'})
        db.soft_remove(doc_ids=[1])

        # Get deleted document without metadata (default)
        deleted_docs = db.deleted()

        assert len(deleted_docs) == 1
        doc = deleted_docs[0]
        assert DELETED_AT_KEY not in doc
        assert doc == {'name': 'test'}


class TestIncludeMetadataParameter:
    """Tests for include_metadata parameter on various methods."""

    def test_all_with_include_metadata(self):
        """Verify all() returns metadata when include_metadata=True."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'test1'})
        db.insert({'name': 'test2'})

        # Without metadata
        docs_no_meta = db.all()
        for doc in docs_no_meta:
            assert CREATED_AT_KEY not in doc

        # With metadata
        docs_with_meta = db.all(include_metadata=True)
        for doc in docs_with_meta:
            assert CREATED_AT_KEY in doc

    def test_search_with_include_metadata(self):
        """Verify search() returns metadata when include_metadata=True."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'test', 'type': 'user'})

        # Without metadata
        docs_no_meta = db.search(where('type') == 'user')
        assert CREATED_AT_KEY not in docs_no_meta[0]

        # With metadata
        docs_with_meta = db.search(where('type') == 'user', include_metadata=True)
        assert CREATED_AT_KEY in docs_with_meta[0]

    def test_search_iter_with_include_metadata(self):
        """Verify search_iter() returns metadata when include_metadata=True."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'test', 'type': 'user'})

        # Without metadata
        for doc in db.search_iter(where('type') == 'user'):
            assert CREATED_AT_KEY not in doc

        # With metadata
        for doc in db.search_iter(where('type') == 'user', include_metadata=True):
            assert CREATED_AT_KEY in doc

    def test_get_with_include_metadata(self):
        """Verify get() returns metadata when include_metadata=True."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'test'})

        # Without metadata
        doc_no_meta = db.get(doc_id=1)
        assert CREATED_AT_KEY not in doc_no_meta

        # With metadata
        doc_with_meta = db.get(doc_id=1, include_metadata=True)
        assert CREATED_AT_KEY in doc_with_meta

    def test_get_multiple_with_include_metadata(self):
        """Verify get() with multiple IDs returns metadata."""
        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple([{'name': 'test1'}, {'name': 'test2'}])

        # Without metadata
        docs_no_meta = db.get(doc_ids=[1, 2])
        for doc in docs_no_meta:
            assert CREATED_AT_KEY not in doc

        # With metadata
        docs_with_meta = db.get(doc_ids=[1, 2], include_metadata=True)
        for doc in docs_with_meta:
            assert CREATED_AT_KEY in doc


class TestUpdatedSince:
    """Tests for updated_since() method."""

    def test_updated_since_returns_updated_docs(self):
        """Verify updated_since returns only documents updated after timestamp."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'doc1'})
        db.insert({'name': 'doc2'})
        db.insert({'name': 'doc3'})

        time.sleep(0.01)
        checkpoint = time.time()
        time.sleep(0.01)

        # Update only doc2
        db.update({'name': 'doc2_updated'}, where('name') == 'doc2')

        # Get documents updated since checkpoint
        updated = db.updated_since(checkpoint)

        assert len(updated) == 1
        assert updated[0]['name'] == 'doc2_updated'
        assert UPDATED_AT_KEY in updated[0]

    def test_updated_since_returns_empty_if_none_updated(self):
        """Verify updated_since returns empty list if no updates."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'doc1'})

        checkpoint = time.time()

        updated = db.updated_since(checkpoint)

        assert updated == []

    def test_updated_since_with_condition(self):
        """Verify updated_since respects both timestamp AND additional conditions.

        This test ensures that:
        1. The timestamp filter actually works (not just the condition)
        2. The condition filter actually works (not just the timestamp)
        By having documents that fail each filter independently.
        """
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'user1', 'type': 'user'})
        db.insert({'name': 'admin1', 'type': 'admin'})
        db.insert({'name': 'user2', 'type': 'user'})

        # Update user1 BEFORE the checkpoint (should be excluded by timestamp)
        db.update({'updated': True}, where('name') == 'user1')

        time.sleep(0.01)
        checkpoint = time.time()
        time.sleep(0.01)

        # Update admin1 AFTER the checkpoint (should be excluded by condition)
        db.update({'updated': True}, where('name') == 'admin1')

        # Update user2 AFTER the checkpoint (should be included - passes both)
        db.update({'updated': True}, where('name') == 'user2')

        # Get only user documents updated since checkpoint
        updated = db.updated_since(checkpoint, where('type') == 'user')

        # Should only return user2:
        # - user1 is excluded because it was updated BEFORE checkpoint
        # - admin1 is excluded because it doesn't match type='user'
        # - user2 passes both: updated after checkpoint AND type='user'
        assert len(updated) == 1, (
            f"Expected 1 document, got {len(updated)}. "
            "This verifies both timestamp and condition filters work together."
        )
        assert updated[0]['name'] == 'user2'
        assert updated[0]['type'] == 'user'
        assert UPDATED_AT_KEY in updated[0]
        assert updated[0][UPDATED_AT_KEY] >= checkpoint

    def test_updated_since_timestamp_filtering_works(self):
        """Verify updated_since actually filters by timestamp, not just returning all updated docs.

        This is a focused test to prove the timestamp logic works correctly.
        """
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'doc1'})
        db.insert({'name': 'doc2'})

        # Update doc1 BEFORE checkpoint
        db.update({'value': 'old_update'}, where('name') == 'doc1')
        doc1_update_time = db.get(where('name') == 'doc1', include_metadata=True)[UPDATED_AT_KEY]

        time.sleep(0.01)
        checkpoint = time.time()
        time.sleep(0.01)

        # Update doc2 AFTER checkpoint
        db.update({'value': 'new_update'}, where('name') == 'doc2')
        doc2_update_time = db.get(where('name') == 'doc2', include_metadata=True)[UPDATED_AT_KEY]

        # Verify our test setup: doc1 was updated before checkpoint, doc2 after
        assert doc1_update_time < checkpoint, "Test setup: doc1 should be updated before checkpoint"
        assert doc2_update_time > checkpoint, "Test setup: doc2 should be updated after checkpoint"

        # Now test updated_since
        updated = db.updated_since(checkpoint)

        # Should only return doc2 (updated after checkpoint)
        assert len(updated) == 1, (
            f"Expected only doc2, got {len(updated)} docs. "
            "If this fails, the timestamp filtering is broken."
        )
        assert updated[0]['name'] == 'doc2'


class TestCreatedSince:
    """Tests for created_since() method."""

    def test_created_since_returns_new_docs(self):
        """Verify created_since returns only documents created after timestamp."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'old_doc'})

        time.sleep(0.01)
        checkpoint = time.time()
        time.sleep(0.01)

        db.insert({'name': 'new_doc'})

        # Get documents created since checkpoint
        created = db.created_since(checkpoint)

        assert len(created) == 1
        assert created[0]['name'] == 'new_doc'
        assert CREATED_AT_KEY in created[0]

    def test_created_since_returns_empty_if_none_new(self):
        """Verify created_since returns empty list if no new documents."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'doc1'})

        time.sleep(0.01)
        checkpoint = time.time()

        created = db.created_since(checkpoint)

        assert created == []

    def test_created_since_with_condition(self):
        """Verify created_since respects additional conditions."""
        db = TinyDB(storage=MemoryStorage)

        checkpoint = time.time()
        time.sleep(0.01)

        db.insert({'name': 'user1', 'type': 'user'})
        db.insert({'name': 'admin1', 'type': 'admin'})

        # Get only user documents created since checkpoint
        created = db.created_since(checkpoint, where('type') == 'user')

        assert len(created) == 1
        assert created[0]['type'] == 'user'


class TestDeletedSince:
    """Tests for deleted_since() method."""

    def test_deleted_since_returns_recently_deleted(self):
        """Verify deleted_since returns documents deleted after timestamp."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'doc1'})
        db.insert({'name': 'doc2'})

        # Delete doc1 early
        db.soft_remove(where('name') == 'doc1')

        time.sleep(0.01)
        checkpoint = time.time()
        time.sleep(0.01)

        # Delete doc2 after checkpoint
        db.soft_remove(where('name') == 'doc2')

        # Get documents deleted since checkpoint
        deleted = db.deleted_since(checkpoint)

        assert len(deleted) == 1
        assert deleted[0]['name'] == 'doc2'
        assert DELETED_AT_KEY in deleted[0]

    def test_deleted_since_returns_empty_if_none_deleted(self):
        """Verify deleted_since returns empty list if no deletions."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'doc1'})

        checkpoint = time.time()

        deleted = db.deleted_since(checkpoint)

        assert deleted == []


class TestChangedSince:
    """Tests for changed_since() method."""

    def test_changed_since_returns_created_and_updated(self):
        """Verify changed_since returns both created and updated docs."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'old_doc'})
        db.insert({'name': 'to_update'})

        time.sleep(0.01)
        checkpoint = time.time()
        time.sleep(0.01)

        # Create a new doc
        db.insert({'name': 'new_doc'})

        # Update existing doc
        db.update({'name': 'updated_doc'}, where('name') == 'to_update')

        # Get all changed documents
        changed = db.changed_since(checkpoint)

        assert len(changed) == 2
        names = {doc['name'] for doc in changed}
        assert 'new_doc' in names
        assert 'updated_doc' in names

    def test_changed_since_excludes_unchanged(self):
        """Verify changed_since excludes documents not changed."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'unchanged'})

        time.sleep(0.01)
        checkpoint = time.time()
        time.sleep(0.01)

        db.insert({'name': 'new_doc'})

        changed = db.changed_since(checkpoint)

        assert len(changed) == 1
        assert changed[0]['name'] == 'new_doc'


class TestMetadataNotAffectingUserData:
    """Tests ensuring metadata doesn't interfere with user data."""

    def test_user_data_unchanged_after_insert(self):
        """Verify user data is unchanged by insert operation."""
        db = TinyDB(storage=MemoryStorage)
        original_data = {'name': 'test', 'value': 123}
        db.insert(original_data)

        doc = db.get(doc_id=1)

        assert doc == original_data

    def test_user_data_unchanged_after_update(self):
        """Verify user data is unchanged by update operation."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'original'})
        db.update({'name': 'updated', 'extra': 'field'}, doc_ids=[1])

        doc = db.get(doc_id=1)

        assert doc == {'name': 'updated', 'extra': 'field'}

    def test_search_condition_not_affected_by_metadata(self):
        """Verify search conditions work correctly without metadata fields."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'test', 'type': 'user'})

        # Search should work normally
        results = db.search(where('type') == 'user')

        assert len(results) == 1
        assert results[0] == {'name': 'test', 'type': 'user'}

    def test_count_not_affected_by_metadata(self):
        """Verify count works correctly."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'type': 'user'})
        db.insert({'type': 'user'})
        db.insert({'type': 'admin'})

        assert db.count(where('type') == 'user') == 2

    def test_len_not_affected_by_metadata(self):
        """Verify len() works correctly."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'test1'})
        db.insert({'name': 'test2'})

        assert len(db) == 2


class TestTimestampPersistence:
    """Tests for timestamp persistence across database operations."""

    def test_timestamps_persist_in_json_storage(self, tmp_path):
        """Verify timestamps are persisted in JSON storage."""
        db_path = tmp_path / 'test.db'

        # Create and close database
        db1 = TinyDB(str(db_path))
        doc_id = db1.insert({'name': 'test'})
        doc1 = db1.get(doc_id=doc_id, include_metadata=True)
        created_at = doc1[CREATED_AT_KEY]
        db1.close()

        # Reopen and verify
        db2 = TinyDB(str(db_path))
        doc2 = db2.get(doc_id=doc_id, include_metadata=True)
        db2.close()

        assert doc2[CREATED_AT_KEY] == created_at

    def test_update_timestamp_persists(self, tmp_path):
        """Verify update timestamps persist."""
        db_path = tmp_path / 'test.db'

        # Create, update, and close
        db1 = TinyDB(str(db_path))
        doc_id = db1.insert({'name': 'original'})
        time.sleep(0.01)
        db1.update({'name': 'updated'}, doc_ids=[doc_id])
        doc1 = db1.get(doc_id=doc_id, include_metadata=True)
        updated_at = doc1[UPDATED_AT_KEY]
        db1.close()

        # Reopen and verify
        db2 = TinyDB(str(db_path))
        doc2 = db2.get(doc_id=doc_id, include_metadata=True)
        db2.close()

        assert doc2[UPDATED_AT_KEY] == updated_at


class TestEdgeCases:
    """Edge case tests for timestamp functionality."""

    def test_upsert_insert_sets_created_at(self):
        """Verify upsert (insert case) sets _created_at."""
        db = TinyDB(storage=MemoryStorage)

        before = time.time()
        db.upsert({'name': 'test', 'type': 'user'}, where('type') == 'user')
        after = time.time()

        doc = db.get(where('type') == 'user', include_metadata=True)

        assert CREATED_AT_KEY in doc
        assert before <= doc[CREATED_AT_KEY] <= after

    def test_upsert_update_sets_updated_at(self):
        """Verify upsert (update case) sets _updated_at."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'original', 'type': 'user'})

        time.sleep(0.01)

        before = time.time()
        db.upsert({'name': 'updated', 'type': 'user'}, where('type') == 'user')
        after = time.time()

        doc = db.get(where('type') == 'user', include_metadata=True)

        assert UPDATED_AT_KEY in doc
        assert before <= doc[UPDATED_AT_KEY] <= after

    def test_restore_does_not_add_new_created_at(self):
        """Verify restore keeps original _created_at."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'test'})

        doc_before = db.get(doc_id=1, include_metadata=True)
        original_created_at = doc_before[CREATED_AT_KEY]

        # Soft delete and restore
        db.soft_remove(doc_ids=[1])
        time.sleep(0.01)
        db.restore(doc_ids=[1])

        doc_after = db.get(doc_id=1, include_metadata=True)

        assert doc_after[CREATED_AT_KEY] == original_created_at

    def test_empty_table_timestamp_queries(self):
        """Verify timestamp queries work on empty tables."""
        db = TinyDB(storage=MemoryStorage)

        assert db.updated_since(0) == []
        assert db.created_since(0) == []
        assert db.deleted_since(0) == []
        assert db.changed_since(0) == []

    def test_future_timestamp_returns_empty(self):
        """Verify queries with future timestamps return empty."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'test'})

        future = time.time() + 86400  # 1 day in future

        assert db.created_since(future) == []
        assert db.changed_since(future) == []
