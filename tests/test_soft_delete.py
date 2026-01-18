"""
Tests for soft delete functionality.
"""

import pytest

from tinydb import TinyDB, where, SOFT_DELETE_KEY
from tinydb.storages import MemoryStorage
from tinydb.table import Document


def test_soft_remove_by_condition(db: TinyDB):
    """Test soft deleting documents by condition."""
    # Soft delete documents where char == 'b'
    removed_ids = db.soft_remove(where('char') == 'b')

    assert len(removed_ids) == 1
    # Document should be hidden from normal queries
    assert len(db) == 2
    assert db.count(where('int') == 1) == 2
    assert db.get(where('char') == 'b') is None


def test_soft_remove_by_doc_ids(db: TinyDB):
    """Test soft deleting documents by document IDs."""
    removed_ids = db.soft_remove(doc_ids=[1, 2])

    assert removed_ids == [1, 2]
    assert len(db) == 1
    assert db.get(doc_id=1) is None
    assert db.get(doc_id=2) is None


def test_soft_remove_invalid(db: TinyDB):
    """Test that soft_remove raises error when no arguments provided."""
    with pytest.raises(RuntimeError):
        db.soft_remove()


def test_restore_by_doc_ids(db: TinyDB):
    """Test restoring soft-deleted documents by IDs."""
    # Soft delete first
    db.soft_remove(doc_ids=[1, 2])
    assert len(db) == 1

    # Restore one document
    restored_ids = db.restore(doc_ids=[1])
    assert restored_ids == [1]
    assert len(db) == 2
    assert db.get(doc_id=1) is not None
    assert db.get(doc_id=2) is None  # Still deleted


def test_restore_by_condition(db: TinyDB):
    """Test restoring soft-deleted documents by condition."""
    # Soft delete all
    db.soft_remove(where('int') == 1)
    assert len(db) == 0

    # Restore only those with char == 'a'
    restored_ids = db.restore(where('char') == 'a')
    assert len(restored_ids) == 1
    assert len(db) == 1
    assert db.get(where('char') == 'a') is not None


def test_restore_all(db: TinyDB):
    """Test restoring all soft-deleted documents."""
    # Soft delete all
    db.soft_remove(where('int') == 1)
    assert len(db) == 0

    # Restore all
    restored_ids = db.restore()
    assert len(restored_ids) == 3
    assert len(db) == 3


def test_deleted_method(db: TinyDB):
    """Test getting list of soft-deleted documents."""
    # Initially no deleted documents
    assert db.deleted() == []

    # Soft delete some documents
    db.soft_remove(where('char') == 'b')

    deleted = db.deleted()
    assert len(deleted) == 1
    assert deleted[0]['char'] == 'b'
    # _deleted field should NOT be visible to users
    assert SOFT_DELETE_KEY not in deleted[0]


def test_deleted_with_condition(db: TinyDB):
    """Test getting soft-deleted documents with a condition."""
    # Soft delete all
    db.soft_remove(where('int') == 1)

    # Get deleted with specific condition
    deleted = db.deleted(where('char') == 'a')
    assert len(deleted) == 1
    assert deleted[0]['char'] == 'a'


def test_deleted_with_pagination(db: TinyDB):
    """Test pagination on deleted documents."""
    # Soft delete all
    db.soft_remove(where('int') == 1)

    # Test limit
    deleted = db.deleted(limit=2)
    assert len(deleted) == 2

    # Test skip
    deleted = db.deleted(skip=1)
    assert len(deleted) == 2

    # Test combined
    deleted = db.deleted(limit=1, skip=1)
    assert len(deleted) == 1


def test_purge_by_doc_ids(db: TinyDB):
    """Test permanently removing soft-deleted documents by IDs."""
    # Soft delete first
    db.soft_remove(doc_ids=[1, 2])

    # Purge one document
    purged_ids = db.purge(doc_ids=[1])
    assert purged_ids == [1]

    # Document 1 should be gone completely
    assert db.get(doc_id=1, include_deleted=True) is None
    # Document 2 should still be in deleted state
    assert db.get(doc_id=2, include_deleted=True) is not None


def test_purge_by_condition(db: TinyDB):
    """Test permanently removing soft-deleted documents by condition."""
    # Soft delete all
    db.soft_remove(where('int') == 1)

    # Purge only those with char == 'a'
    purged_ids = db.purge(where('char') == 'a')
    assert len(purged_ids) == 1

    # Check that 'a' is completely gone
    deleted = db.deleted()
    assert len(deleted) == 2
    assert all(d['char'] != 'a' for d in deleted)


def test_purge_all(db: TinyDB):
    """Test purging all soft-deleted documents."""
    # Soft delete all
    db.soft_remove(where('int') == 1)

    # Purge all
    purged_ids = db.purge()
    assert len(purged_ids) == 3

    # All documents should be gone
    assert db.deleted() == []
    assert len(db.all(include_deleted=True)) == 0


def test_purge_non_deleted_does_nothing(db: TinyDB):
    """Test that purge only affects soft-deleted documents."""
    # Try to purge non-deleted documents
    purged_ids = db.purge(doc_ids=[1, 2])
    assert purged_ids == []
    assert len(db) == 3


def test_include_deleted_in_search(db: TinyDB):
    """Test including deleted documents in search results."""
    db.soft_remove(where('char') == 'b')

    # Normal search excludes deleted
    results = db.search(where('int') == 1)
    assert len(results) == 2

    # With include_deleted=True
    results = db.search(where('int') == 1, include_deleted=True)
    assert len(results) == 3


def test_include_deleted_in_get(db: TinyDB):
    """Test including deleted documents in get."""
    db.soft_remove(doc_ids=[1])

    # Normal get excludes deleted
    assert db.get(doc_id=1) is None

    # With include_deleted=True
    doc = db.get(doc_id=1, include_deleted=True)
    assert doc is not None
    assert doc['char'] == 'a'


def test_include_deleted_in_all(db: TinyDB):
    """Test including deleted documents in all."""
    db.soft_remove(where('char') == 'b')

    # Normal all excludes deleted
    assert len(db.all()) == 2

    # With include_deleted=True
    assert len(db.all(include_deleted=True)) == 3


def test_include_deleted_in_contains(db: TinyDB):
    """Test including deleted documents in contains check."""
    db.soft_remove(doc_ids=[1])

    # Normal contains excludes deleted
    assert not db.contains(doc_id=1)

    # With include_deleted=True
    assert db.contains(doc_id=1, include_deleted=True)


def test_include_deleted_in_count(db: TinyDB):
    """Test including deleted documents in count."""
    db.soft_remove(where('char') == 'b')

    # Normal count excludes deleted
    assert db.count(where('int') == 1) == 2

    # With include_deleted=True
    assert db.count(where('int') == 1, include_deleted=True) == 3


def test_soft_delete_preserves_document_data(db: TinyDB):
    """Test that soft delete preserves original document data."""
    db.soft_remove(doc_ids=[1])

    # Check the deleted document still has its data
    deleted = db.deleted()
    assert len(deleted) == 1
    doc = deleted[0]
    assert doc['int'] == 1
    assert doc['char'] == 'a'
    assert doc.doc_id == 1
    # _deleted field should NOT be visible to users
    assert SOFT_DELETE_KEY not in doc


def test_soft_delete_already_deleted(db: TinyDB):
    """Test that soft deleting an already deleted doc doesn't double-delete."""
    db.soft_remove(doc_ids=[1])
    db.soft_remove(doc_ids=[1])

    # Should only be one deleted document with ID 1
    deleted = db.deleted()
    assert sum(1 for d in deleted if d.doc_id == 1) == 1


def test_update_does_not_affect_deleted(db: TinyDB):
    """Test that update operations don't affect soft-deleted documents."""
    db.soft_remove(doc_ids=[1])

    # Update all non-deleted documents
    db.update({'updated': True})

    # Restore and check the deleted document wasn't updated
    db.restore(doc_ids=[1])
    doc = db.get(doc_id=1)
    assert 'updated' not in doc


def test_soft_delete_with_json_storage(tmpdir):
    """Test that soft delete works with JSON storage persistence."""
    path = str(tmpdir.join('test.db'))

    # Create db and soft delete
    with TinyDB(path) as db:
        db.insert({'name': 'Alice'})
        db.insert({'name': 'Bob'})
        db.soft_remove(where('name') == 'Bob')
        assert len(db) == 1

    # Reopen and verify
    with TinyDB(path) as db:
        assert len(db) == 1
        assert db.get(where('name') == 'Alice') is not None
        assert db.get(where('name') == 'Bob') is None

        # Deleted should still be there
        deleted = db.deleted()
        assert len(deleted) == 1
        assert deleted[0]['name'] == 'Bob'

        # Restore and verify
        db.restore()
        assert len(db) == 2


def test_len_excludes_deleted(db: TinyDB):
    """Test that len() excludes soft-deleted documents."""
    assert len(db) == 3
    db.soft_remove(doc_ids=[1])
    assert len(db) == 2


def test_iter_excludes_deleted(db: TinyDB):
    """Test that iterating the table excludes soft-deleted documents."""
    db.soft_remove(doc_ids=[1])
    docs = list(db)
    assert len(docs) == 2
    assert all(d.doc_id != 1 for d in docs)


def test_soft_delete_key_not_visible_in_restored(db: TinyDB):
    """Test that the _deleted key is removed after restore."""
    db.soft_remove(doc_ids=[1])
    db.restore(doc_ids=[1])

    doc = db.get(doc_id=1)
    assert SOFT_DELETE_KEY not in doc


def test_search_cache_excludes_deleted(db: TinyDB):
    """Test that search cache properly excludes deleted documents."""
    query = where('int') == 1

    # First search populates cache
    results1 = db.search(query)
    assert len(results1) == 3

    # Soft delete
    db.soft_remove(doc_ids=[1])

    # Cache should be cleared, new search should exclude deleted
    results2 = db.search(query)
    assert len(results2) == 2


def test_get_multiple_ids_excludes_deleted(db: TinyDB):
    """Test getting multiple documents by IDs excludes deleted."""
    db.soft_remove(doc_ids=[2])

    # Get multiple should exclude deleted
    docs = db.get(doc_ids=[1, 2, 3])
    assert len(docs) == 2
    assert all(d.doc_id != 2 for d in docs)

    # With include_deleted should include all
    docs = db.get(doc_ids=[1, 2, 3], include_deleted=True)
    assert len(docs) == 3


def test_deleted_condition_does_not_see_deleted_field(db: TinyDB):
    """Test that conditions on deleted() don't see the _deleted field."""
    db.soft_remove(where('int') == 1)

    # This query should work - searching by original fields
    deleted = db.deleted(where('char') == 'a')
    assert len(deleted) == 1

    # This query should NOT find anything because _deleted is hidden
    deleted_with_flag = db.deleted(where('_deleted') == True)
    assert len(deleted_with_flag) == 0


def test_restore_condition_does_not_see_deleted_field(db: TinyDB):
    """Test that conditions on restore() don't see the _deleted field."""
    db.soft_remove(where('int') == 1)

    # This should work - restore by original field
    restored = db.restore(where('char') == 'a')
    assert len(restored) == 1
    assert len(db) == 1  # One doc restored

    # Trying to restore by _deleted field should not match anything
    db.soft_remove(where('char') == 'b')
    restored_by_flag = db.restore(where('_deleted') == True)
    assert len(restored_by_flag) == 0


def test_purge_condition_does_not_see_deleted_field(db: TinyDB):
    """Test that conditions on purge() don't see the _deleted field."""
    db.soft_remove(where('int') == 1)

    # This should work - purge by original field
    purged = db.purge(where('char') == 'a')
    assert len(purged) == 1

    # Trying to purge by _deleted field should not match anything
    purged_by_flag = db.purge(where('_deleted') == True)
    assert len(purged_by_flag) == 0


def test_include_deleted_strips_deleted_field(db: TinyDB):
    """Test that include_deleted=True still strips the _deleted field."""
    db.soft_remove(doc_ids=[1])

    # Get all docs including deleted
    all_docs = db.all(include_deleted=True)
    assert len(all_docs) == 3

    # None of them should have the _deleted field visible
    for doc in all_docs:
        assert SOFT_DELETE_KEY not in doc

    # Same for search with include_deleted
    results = db.search(where('int') == 1, include_deleted=True)
    assert len(results) == 3
    for doc in results:
        assert SOFT_DELETE_KEY not in doc

    # Same for get with include_deleted
    doc = db.get(doc_id=1, include_deleted=True)
    assert doc is not None
    assert SOFT_DELETE_KEY not in doc

    # Same for get with doc_ids and include_deleted
    docs = db.get(doc_ids=[1, 2], include_deleted=True)
    assert len(docs) == 2
    for doc in docs:
        assert SOFT_DELETE_KEY not in doc


def test_search_iter_excludes_soft_deleted(db: TinyDB):
    """Test that search_iter excludes soft-deleted documents by default."""
    db.soft_remove(where('char') == 'b')

    # Normal search_iter excludes deleted
    results = list(db.search_iter(where('int') == 1))
    assert len(results) == 2
    assert all(doc['char'] != 'b' for doc in results)


def test_search_iter_include_deleted(db: TinyDB):
    """Test that search_iter can include soft-deleted documents."""
    db.soft_remove(where('char') == 'b')

    # With include_deleted=True
    results = list(db.search_iter(where('int') == 1, include_deleted=True))
    assert len(results) == 3

    # _deleted field should NOT be visible to users
    for doc in results:
        assert SOFT_DELETE_KEY not in doc


def test_search_iter_include_deleted_with_pagination(db: TinyDB):
    """Test search_iter with include_deleted and pagination."""
    db.soft_remove(where('char') == 'b')

    # With include_deleted=True and limit
    results = list(db.search_iter(where('int') == 1, include_deleted=True, limit=2))
    assert len(results) == 2

    # With include_deleted=True and skip
    results = list(db.search_iter(where('int') == 1, include_deleted=True, skip=1))
    assert len(results) == 2

    # With include_deleted=True and both limit and skip
    results = list(db.search_iter(where('int') == 1, include_deleted=True, limit=1, skip=1))
    assert len(results) == 1


def test_search_iter_matches_search_with_soft_delete(db: TinyDB):
    """Test that search_iter returns same results as search with soft delete."""
    db.soft_remove(where('char') == 'b')

    query = where('int') == 1

    # Compare normal results
    search_results = db.search(query)
    iter_results = list(db.search_iter(query))
    assert search_results == iter_results

    # Compare with include_deleted
    search_results = db.search(query, include_deleted=True)
    iter_results = list(db.search_iter(query, include_deleted=True))
    assert search_results == iter_results
