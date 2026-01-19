"""
Tests for the query profiling feature.
"""

import pytest
from tinydb import TinyDB, where, Query
from tinydb.storages import MemoryStorage
from tinydb.profiling import QueryProfiler, QueryStats, QuerySummary


@pytest.fixture
def db():
    """Create a fresh in-memory database for each test."""
    db_ = TinyDB(storage=MemoryStorage)
    db_.drop_tables()
    return db_


@pytest.fixture
def populated_db(db):
    """Create a database with sample data."""
    # Insert test data
    db.insert_multiple([
        {'name': 'Alice', 'age': 30, 'city': 'NYC'},
        {'name': 'Bob', 'age': 25, 'city': 'LA'},
        {'name': 'Charlie', 'age': 35, 'city': 'NYC'},
        {'name': 'Diana', 'age': 28, 'city': 'Chicago'},
        {'name': 'Eve', 'age': 32, 'city': 'LA'},
    ])
    return db


class TestQueryProfilerBasics:
    """Test basic QueryProfiler functionality."""

    def test_profiler_creation(self):
        """Test creating a QueryProfiler instance."""
        profiler = QueryProfiler()
        assert profiler.enabled is True
        assert len(profiler.get_stats()) == 0

    def test_profiler_enable_disable(self):
        """Test enabling and disabling the profiler."""
        profiler = QueryProfiler()
        assert profiler.enabled is True

        profiler.enabled = False
        assert profiler.enabled is False

        profiler.enabled = True
        assert profiler.enabled is True

    def test_profiler_clear(self):
        """Test clearing profiler statistics."""
        profiler = QueryProfiler()
        profiler.record_query(
            table_name='test',
            query=where('x') == 1,
            execution_time_ms=10.0,
            docs_scanned=100,
            docs_matched=5
        )
        assert len(profiler.get_stats()) == 1

        profiler.clear()
        assert len(profiler.get_stats()) == 0

    def test_profiler_repr(self):
        """Test profiler string representation."""
        profiler = QueryProfiler()
        repr_str = repr(profiler)
        assert 'QueryProfiler' in repr_str
        assert 'enabled' in repr_str


class TestDatabaseProfilingIntegration:
    """Test profiling integration with TinyDB."""

    def test_enable_profiling(self, db):
        """Test enabling profiling on a database."""
        profiler = db.enable_profiling()
        assert db.profiler is profiler
        assert isinstance(profiler, QueryProfiler)

    def test_enable_profiling_with_custom_profiler(self, db):
        """Test enabling profiling with a custom profiler."""
        custom_profiler = QueryProfiler()
        result = db.enable_profiling(custom_profiler)
        assert result is custom_profiler
        assert db.profiler is custom_profiler

    def test_disable_profiling(self, db):
        """Test disabling profiling."""
        profiler = db.enable_profiling()
        db.disable_profiling()
        assert db.profiler is None

    def test_profiler_persists_after_disable(self, populated_db):
        """Test that profiler data persists after disabling."""
        profiler = populated_db.enable_profiling()

        # Perform queries
        populated_db.search(where('name') == 'Alice')
        populated_db.search(where('age') > 25)

        # Disable profiling
        populated_db.disable_profiling()

        # Profiler should still have the data
        assert len(profiler.get_stats()) == 2

    def test_profiling_existing_tables(self, populated_db):
        """Test that enabling profiling affects existing tables."""
        # Access a table before enabling profiling
        users = populated_db.table('users')
        users.insert({'name': 'Test'})

        # Enable profiling
        profiler = populated_db.enable_profiling()

        # Queries on existing table should be profiled
        users.search(where('name') == 'Test')

        stats = profiler.get_stats('users')
        assert len(stats) == 1


class TestSearchProfiling:
    """Test profiling of search operations."""

    def test_search_profiling(self, populated_db):
        """Test that search operations are profiled."""
        profiler = populated_db.enable_profiling()

        # Perform a search
        results = populated_db.search(where('name') == 'Alice')
        assert len(results) == 1

        # Check profiling data
        stats = profiler.get_stats()
        assert len(stats) == 1

        stat = stats[0]
        assert stat.call_count == 1
        assert stat.total_time_ms > 0
        assert stat.documents_scanned == 5
        assert stat.documents_matched == 1
        assert stat.table_name == '_default'

    def test_multiple_search_same_query(self, populated_db):
        """Test profiling multiple executions of the same query."""
        profiler = populated_db.enable_profiling()

        query = where('city') == 'NYC'

        # Perform the same query multiple times
        populated_db.search(query)
        # Clear cache to force re-execution
        populated_db.clear_cache()
        populated_db.search(query)
        populated_db.clear_cache()
        populated_db.search(query)

        stats = profiler.get_stats()
        assert len(stats) == 1

        stat = stats[0]
        assert stat.call_count == 3
        assert stat.documents_matched == 6  # 2 matches * 3 calls

    def test_different_queries_profiled_separately(self, populated_db):
        """Test that different queries are tracked separately."""
        profiler = populated_db.enable_profiling()

        populated_db.search(where('name') == 'Alice')
        populated_db.search(where('age') > 30)
        populated_db.search(where('city') == 'LA')

        stats = profiler.get_stats()
        assert len(stats) == 3

    def test_search_with_limit_skip(self, populated_db):
        """Test profiling with pagination parameters."""
        profiler = populated_db.enable_profiling()

        populated_db.search(where('age') > 20, limit=2, skip=1)

        stats = profiler.get_stats()
        assert len(stats) == 1
        # All documents should be scanned, but only matching ones counted
        assert stats[0].documents_scanned == 5


class TestSearchIterProfiling:
    """Test profiling of search_iter operations."""

    def test_search_iter_profiling(self, populated_db):
        """Test that search_iter operations are profiled."""
        profiler = populated_db.enable_profiling()

        # Consume the iterator
        results = list(populated_db.search_iter(where('city') == 'NYC'))
        assert len(results) == 2

        stats = profiler.get_stats()
        assert len(stats) == 1
        assert stats[0].documents_matched == 2

    def test_search_iter_with_limit(self, populated_db):
        """Test search_iter profiling with early exit due to limit."""
        profiler = populated_db.enable_profiling()

        results = list(populated_db.search_iter(where('age') > 20, limit=2))
        assert len(results) == 2

        stats = profiler.get_stats()
        assert len(stats) == 1
        assert stats[0].documents_matched == 2


class TestGetProfiling:
    """Test profiling of get operations."""

    def test_get_with_condition_profiling(self, populated_db):
        """Test that get with condition is profiled."""
        profiler = populated_db.enable_profiling()

        result = populated_db.get(where('name') == 'Bob')
        assert result is not None
        assert result['name'] == 'Bob'

        stats = profiler.get_stats()
        assert len(stats) == 1
        assert stats[0].documents_matched == 1

    def test_get_no_match_profiling(self, populated_db):
        """Test profiling when get finds no match."""
        profiler = populated_db.enable_profiling()

        result = populated_db.get(where('name') == 'NonExistent')
        assert result is None

        stats = profiler.get_stats()
        assert len(stats) == 1
        assert stats[0].documents_matched == 0
        assert stats[0].documents_scanned == 5

    def test_get_by_doc_id_not_profiled(self, populated_db):
        """Test that get by doc_id is not profiled (no query involved)."""
        profiler = populated_db.enable_profiling()

        # Get by doc_id doesn't involve a query
        populated_db.get(doc_id=1)

        stats = profiler.get_stats()
        assert len(stats) == 0


class TestMultiTableProfiling:
    """Test profiling across multiple tables."""

    def test_profile_multiple_tables(self, db):
        """Test that queries on different tables are tracked separately."""
        profiler = db.enable_profiling()

        # Create and query multiple tables
        users = db.table('users')
        orders = db.table('orders')

        users.insert_multiple([
            {'name': 'Alice', 'email': 'alice@example.com'},
            {'name': 'Bob', 'email': 'bob@example.com'},
        ])

        orders.insert_multiple([
            {'order_id': 1, 'user': 'Alice', 'total': 100},
            {'order_id': 2, 'user': 'Bob', 'total': 200},
            {'order_id': 3, 'user': 'Alice', 'total': 150},
        ])

        # Query both tables
        users.search(where('name') == 'Alice')
        orders.search(where('user') == 'Alice')
        orders.search(where('total') > 100)

        # Check stats by table
        user_stats = profiler.get_stats('users')
        order_stats = profiler.get_stats('orders')

        assert len(user_stats) == 1
        assert len(order_stats) == 2

        # Check table names in stats
        assert user_stats[0].table_name == 'users'
        assert all(s.table_name == 'orders' for s in order_stats)

    def test_queries_by_table_in_summary(self, db):
        """Test that summary groups queries by table."""
        profiler = db.enable_profiling()

        users = db.table('users')
        orders = db.table('orders')

        users.insert({'name': 'Test'})
        orders.insert({'order_id': 1})

        users.search(where('name') == 'Test')
        orders.search(where('order_id') == 1)

        summary = profiler.get_summary()
        assert 'users' in summary.queries_by_table
        assert 'orders' in summary.queries_by_table


class TestQueryStats:
    """Test QueryStats class functionality."""

    def test_query_stats_avg_time(self):
        """Test average time calculation."""
        stats = QueryStats(query_repr='test', table_name='test')
        stats.record(10.0)
        stats.record(20.0)
        stats.record(30.0)

        assert stats.call_count == 3
        assert stats.total_time_ms == 60.0
        assert stats.avg_time_ms == 20.0

    def test_query_stats_min_max(self):
        """Test min/max time tracking."""
        stats = QueryStats(query_repr='test', table_name='test')
        stats.record(15.0)
        stats.record(5.0)
        stats.record(25.0)

        assert stats.min_time_ms == 5.0
        assert stats.max_time_ms == 25.0

    def test_query_stats_to_dict(self):
        """Test conversion to dictionary."""
        stats = QueryStats(query_repr='test query', table_name='users')
        stats.record(10.0, docs_scanned=100, docs_matched=5)

        d = stats.to_dict()
        assert d['query'] == 'test query'
        assert d['table'] == 'users'
        assert d['call_count'] == 1
        assert d['documents_scanned'] == 100
        assert d['documents_matched'] == 5


class TestQuerySummary:
    """Test QuerySummary functionality."""

    def test_summary_totals(self, populated_db):
        """Test summary total calculations."""
        profiler = populated_db.enable_profiling()

        populated_db.search(where('name') == 'Alice')
        populated_db.search(where('age') > 25)
        populated_db.search(where('city') == 'NYC')

        summary = profiler.get_summary()
        assert summary.total_queries == 3
        assert summary.unique_queries == 3
        assert summary.total_time_ms > 0

    def test_slowest_queries(self, populated_db):
        """Test that slowest queries are sorted correctly."""
        profiler = populated_db.enable_profiling()

        # Execute multiple queries - profiler tracks them
        populated_db.search(where('name') == 'Alice')
        populated_db.search(where('age') > 25)

        summary = profiler.get_summary(slowest_limit=5)
        assert len(summary.slowest_queries) <= 5

        # Verify sorted by avg_time_ms (descending)
        for i in range(len(summary.slowest_queries) - 1):
            assert summary.slowest_queries[i].avg_time_ms >= \
                   summary.slowest_queries[i + 1].avg_time_ms

    def test_most_called_queries(self, populated_db):
        """Test that most called queries are sorted correctly."""
        profiler = populated_db.enable_profiling()

        query1 = where('name') == 'Alice'
        query2 = where('age') > 25

        # Execute query1 three times, query2 once
        for _ in range(3):
            populated_db.search(query1)
            populated_db.clear_cache()

        populated_db.search(query2)

        summary = profiler.get_summary(most_called_limit=5)

        # First should be the most called
        assert summary.most_called_queries[0].call_count >= \
               summary.most_called_queries[1].call_count

    def test_summary_repr(self, populated_db):
        """Test summary string representation."""
        profiler = populated_db.enable_profiling()
        populated_db.search(where('name') == 'Alice')

        summary = profiler.get_summary()
        summary_str = repr(summary)

        assert 'QUERY PROFILING SUMMARY' in summary_str
        assert 'SLOWEST QUERIES' in summary_str
        assert '_default' in summary_str  # Table name

    def test_summary_to_dict(self, populated_db):
        """Test summary conversion to dictionary."""
        profiler = populated_db.enable_profiling()
        populated_db.search(where('name') == 'Alice')

        summary = profiler.get_summary()
        d = summary.to_dict()

        assert 'total_queries' in d
        assert 'total_time_ms' in d
        assert 'slowest_queries' in d
        assert 'most_called_queries' in d
        assert 'queries_by_table' in d


class TestQueryRepresentation:
    """Test human-readable query representation."""

    def test_equality_query_repr(self, populated_db):
        """Test representation of equality queries."""
        profiler = populated_db.enable_profiling()
        populated_db.search(where('name') == 'Alice')

        stats = profiler.get_stats()[0]
        assert 'name' in stats.query_repr
        assert '==' in stats.query_repr

    def test_comparison_query_repr(self, populated_db):
        """Test representation of comparison queries."""
        profiler = populated_db.enable_profiling()
        populated_db.search(where('age') > 25)

        stats = profiler.get_stats()[0]
        assert 'age' in stats.query_repr
        assert '>' in stats.query_repr

    def test_complex_query_repr(self, populated_db):
        """Test representation of complex AND/OR queries."""
        profiler = populated_db.enable_profiling()

        complex_query = (where('age') > 25) & (where('city') == 'NYC')
        populated_db.search(complex_query)

        stats = profiler.get_stats()[0]
        # Should contain both fields
        assert 'age' in stats.query_repr or '>' in stats.query_repr
        assert '&' in stats.query_repr or 'and' in stats.query_repr.lower()


class TestProfilingWhenDisabled:
    """Test that profiling doesn't affect queries when disabled."""

    def test_no_profiling_by_default(self, populated_db):
        """Test that profiling is disabled by default."""
        assert populated_db.profiler is None

        # Queries should work normally
        results = populated_db.search(where('name') == 'Alice')
        assert len(results) == 1

    def test_disabled_profiler_no_recording(self, populated_db):
        """Test that disabled profiler doesn't record queries."""
        profiler = QueryProfiler()
        profiler.enabled = False

        populated_db.enable_profiling(profiler)
        populated_db.search(where('name') == 'Alice')

        assert len(profiler.get_stats()) == 0


class TestHelperMethods:
    """Test profiler helper methods."""

    def test_get_slowest_queries_with_limit(self, populated_db):
        """Test getting slowest queries with limit."""
        profiler = populated_db.enable_profiling()

        # Execute several queries
        for i in range(10):
            populated_db.search(where('age') > i)

        slowest = profiler.get_slowest_queries(limit=3)
        assert len(slowest) == 3

    def test_get_most_called_queries_with_limit(self, populated_db):
        """Test getting most called queries with limit."""
        profiler = populated_db.enable_profiling()

        # Execute same query multiple times
        query = where('name') == 'Alice'
        for _ in range(5):
            populated_db.search(query)
            populated_db.clear_cache()

        most_called = profiler.get_most_called_queries(limit=1)
        assert len(most_called) == 1
        assert most_called[0].call_count == 5

    def test_get_queries_by_total_time(self, populated_db):
        """Test getting queries sorted by total time."""
        profiler = populated_db.enable_profiling()

        # Execute queries
        populated_db.search(where('name') == 'Alice')
        populated_db.search(where('age') > 20)

        by_total = profiler.get_queries_by_total_time(limit=10)
        assert len(by_total) == 2

        # Verify sorted by total_time_ms (descending)
        for i in range(len(by_total) - 1):
            assert by_total[i].total_time_ms >= by_total[i + 1].total_time_ms

    def test_get_stats_filter_by_table(self, db):
        """Test filtering stats by table name."""
        profiler = db.enable_profiling()

        # Query multiple tables
        users = db.table('users')
        orders = db.table('orders')

        users.insert({'name': 'Test'})
        orders.insert({'id': 1})

        users.search(where('name') == 'Test')
        orders.search(where('id') == 1)

        # Get stats filtered by table
        user_stats = profiler.get_stats('users')
        assert len(user_stats) == 1
        assert user_stats[0].table_name == 'users'

        order_stats = profiler.get_stats('orders')
        assert len(order_stats) == 1
        assert order_stats[0].table_name == 'orders'

        # Get all stats
        all_stats = profiler.get_stats()
        assert len(all_stats) == 2


class TestValidation:
    """Test profiler validation and edge cases."""

    def test_docs_matched_exceeds_docs_scanned_raises_error(self):
        """Test that docs_matched > docs_scanned raises ValueError."""
        profiler = QueryProfiler()

        with pytest.raises(ValueError) as exc_info:
            profiler.record_query(
                table_name='test',
                query=where('x') == 1,
                execution_time_ms=10.0,
                docs_scanned=5,
                docs_matched=10  # Invalid: more matched than scanned
            )

        assert "docs_matched (10) cannot be greater than docs_scanned (5)" in str(exc_info.value)
        assert "test" in str(exc_info.value)  # Table name in error

    def test_docs_matched_equals_docs_scanned_is_valid(self):
        """Test that docs_matched == docs_scanned is valid."""
        profiler = QueryProfiler()

        # Should not raise
        profiler.record_query(
            table_name='test',
            query=where('x') == 1,
            execution_time_ms=10.0,
            docs_scanned=5,
            docs_matched=5  # Valid: all scanned docs matched
        )

        stats = profiler.get_stats()
        assert len(stats) == 1
        assert stats[0].documents_matched == 5

    def test_search_iter_limit_zero_records_query(self, populated_db):
        """Test that search_iter with limit=0 records the query."""
        profiler = populated_db.enable_profiling()

        # Consume the iterator (should be empty)
        results = list(populated_db.search_iter(where('name') == 'Alice', limit=0))
        assert len(results) == 0

        # Query should still be recorded
        stats = profiler.get_stats()
        assert len(stats) == 1
        assert stats[0].documents_matched == 0
        assert stats[0].documents_scanned == 5  # All docs were "scanned" conceptually
        assert stats[0].call_count == 1

    def test_search_iter_docs_matched_with_skip_and_limit(self, populated_db):
        """Test that docs_matched counts only yielded docs after skip/limit."""
        profiler = populated_db.enable_profiling()

        # All 5 docs match age > 20 query, but we skip 2 and limit to 2
        # So only 2 should be yielded and counted as matched
        query = where('age') > 20
        results = list(populated_db.search_iter(query, skip=2, limit=2))
        assert len(results) == 2

        stats = profiler.get_stats()
        assert len(stats) == 1
        assert stats[0].documents_matched == 2  # Only yielded count
        assert stats[0].documents_scanned == 5  # All docs scanned

    def test_search_iter_docs_matched_with_skip_only(self, populated_db):
        """Test docs_matched with skip but no limit."""
        profiler = populated_db.enable_profiling()

        # All 5 docs match age > 20, skip first 3
        query = where('age') > 20
        results = list(populated_db.search_iter(query, skip=3))
        assert len(results) == 2  # 5 total - 3 skipped = 2 yielded

        stats = profiler.get_stats()
        assert len(stats) == 1
        assert stats[0].documents_matched == 2  # Only yielded count
        assert stats[0].documents_scanned == 5

    def test_search_iter_docs_matched_with_limit_only(self, populated_db):
        """Test docs_matched with limit but no skip."""
        profiler = populated_db.enable_profiling()

        # All 5 docs match age > 20, limit to 3
        query = where('age') > 20
        results = list(populated_db.search_iter(query, limit=3))
        assert len(results) == 3

        stats = profiler.get_stats()
        assert len(stats) == 1
        assert stats[0].documents_matched == 3  # Only yielded count
        assert stats[0].documents_scanned == 5

    def test_search_iter_docs_matched_skip_exceeds_matches(self, populated_db):
        """Test docs_matched when skip exceeds total matching documents."""
        profiler = populated_db.enable_profiling()

        # Only 2 docs match city == 'NYC', but we skip 5
        query = where('city') == 'NYC'
        results = list(populated_db.search_iter(query, skip=5))
        assert len(results) == 0  # All matches skipped

        stats = profiler.get_stats()
        assert len(stats) == 1
        assert stats[0].documents_matched == 0  # Nothing yielded
        assert stats[0].documents_scanned == 5


class TestProfilingWithIncludeDeleted:
    """Test profiling consistency with include_deleted parameter."""

    def test_search_profiling_exclude_deleted(self, db):
        """Test profiling when soft-deleted docs are excluded (default)."""
        # Insert docs and soft-delete some
        db.insert_multiple([
            {'name': 'Alice', 'status': 'active'},
            {'name': 'Bob', 'status': 'active'},
            {'name': 'Charlie', 'status': 'active'},
        ])
        db.soft_remove(where('name') == 'Bob')

        profiler = db.enable_profiling()

        # Search excluding deleted (default)
        results = db.search(where('status') == 'active')
        assert len(results) == 2  # Alice and Charlie

        stats = profiler.get_stats()
        assert len(stats) == 1
        assert stats[0].documents_scanned == 2  # Only non-deleted scanned
        assert stats[0].documents_matched == 2

    def test_search_profiling_include_deleted(self, db):
        """Test profiling when soft-deleted docs are included."""
        # Insert docs and soft-delete some
        db.insert_multiple([
            {'name': 'Alice', 'status': 'active'},
            {'name': 'Bob', 'status': 'active'},
            {'name': 'Charlie', 'status': 'active'},
        ])
        db.soft_remove(where('name') == 'Bob')

        profiler = db.enable_profiling()

        # Search including deleted
        results = db.search(where('status') == 'active', include_deleted=True)
        assert len(results) == 3  # All three including soft-deleted Bob

        stats = profiler.get_stats()
        assert len(stats) == 1
        assert stats[0].documents_scanned == 3  # All docs scanned
        assert stats[0].documents_matched == 3

    def test_search_iter_profiling_exclude_deleted(self, db):
        """Test search_iter profiling when soft-deleted docs are excluded."""
        db.insert_multiple([
            {'name': 'Alice', 'age': 30},
            {'name': 'Bob', 'age': 25},
            {'name': 'Charlie', 'age': 35},
        ])
        db.soft_remove(where('name') == 'Charlie')

        profiler = db.enable_profiling()

        results = list(db.search_iter(where('age') > 20))
        assert len(results) == 2  # Alice and Bob

        stats = profiler.get_stats()
        assert len(stats) == 1
        assert stats[0].documents_scanned == 2
        assert stats[0].documents_matched == 2

    def test_search_iter_profiling_include_deleted(self, db):
        """Test search_iter profiling when soft-deleted docs are included."""
        db.insert_multiple([
            {'name': 'Alice', 'age': 30},
            {'name': 'Bob', 'age': 25},
            {'name': 'Charlie', 'age': 35},
        ])
        db.soft_remove(where('name') == 'Charlie')

        profiler = db.enable_profiling()

        results = list(db.search_iter(where('age') > 20, include_deleted=True))
        assert len(results) == 3  # All three

        stats = profiler.get_stats()
        assert len(stats) == 1
        assert stats[0].documents_scanned == 3
        assert stats[0].documents_matched == 3

    def test_get_profiling_exclude_deleted(self, db):
        """Test get profiling when soft-deleted docs are excluded."""
        db.insert_multiple([
            {'name': 'Alice'},
            {'name': 'Bob'},
        ])
        db.soft_remove(where('name') == 'Bob')

        profiler = db.enable_profiling()

        # Try to get deleted doc (should not find)
        result = db.get(where('name') == 'Bob')
        assert result is None

        stats = profiler.get_stats()
        assert len(stats) == 1
        assert stats[0].documents_scanned == 1  # Only Alice scanned
        assert stats[0].documents_matched == 0

    def test_get_profiling_include_deleted(self, db):
        """Test get profiling when soft-deleted docs are included."""
        db.insert_multiple([
            {'name': 'Alice'},
            {'name': 'Bob'},
        ])
        db.soft_remove(where('name') == 'Bob')

        profiler = db.enable_profiling()

        # Get deleted doc with include_deleted=True
        result = db.get(where('name') == 'Bob', include_deleted=True)
        assert result is not None
        assert result['name'] == 'Bob'

        stats = profiler.get_stats()
        assert len(stats) == 1
        assert stats[0].documents_scanned == 2  # Both scanned
        assert stats[0].documents_matched == 1
