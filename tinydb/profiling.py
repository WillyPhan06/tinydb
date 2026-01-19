"""
Query profiling module for TinyDB.

This module provides tools for monitoring and analyzing query performance
in TinyDB. It tracks execution times, call counts, and provides summary
reports to help developers identify slow queries and optimization opportunities.

Usage:

>>> from tinydb import TinyDB
>>> from tinydb.profiling import QueryProfiler
>>>
>>> db = TinyDB('db.json')
>>> profiler = QueryProfiler()
>>> db.enable_profiling(profiler)
>>>
>>> # Perform some queries
>>> db.search(where('name') == 'John')
>>> db.search(where('age') > 25)
>>>
>>> # Get a summary
>>> profiler.get_summary()
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)


__all__ = ('QueryProfiler', 'QueryStats', 'QuerySummary')


@dataclass
class QueryStats:
    """
    Statistics for a single query type.

    Attributes:
        query_repr: String representation of the query
        table_name: Name of the table this query was executed against
        call_count: Number of times this query was executed
        total_time_ms: Total execution time in milliseconds
        min_time_ms: Minimum execution time in milliseconds
        max_time_ms: Maximum execution time in milliseconds
        avg_time_ms: Average execution time in milliseconds
        documents_scanned: Total number of documents scanned
        documents_matched: Total number of documents matched
    """
    query_repr: str
    table_name: str
    call_count: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    documents_scanned: int = 0
    documents_matched: int = 0

    @property
    def avg_time_ms(self) -> float:
        """Calculate the average execution time in milliseconds."""
        if self.call_count == 0:
            return 0.0
        return self.total_time_ms / self.call_count

    def record(
        self,
        execution_time_ms: float,
        docs_scanned: int = 0,
        docs_matched: int = 0
    ) -> None:
        """
        Record a new query execution.

        :param execution_time_ms: Execution time in milliseconds
        :param docs_scanned: Number of documents scanned
        :param docs_matched: Number of documents matched
        """
        self.call_count += 1
        self.total_time_ms += execution_time_ms
        self.min_time_ms = min(self.min_time_ms, execution_time_ms)
        self.max_time_ms = max(self.max_time_ms, execution_time_ms)
        self.documents_scanned += docs_scanned
        self.documents_matched += docs_matched

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy inspection."""
        return {
            'query': self.query_repr,
            'table': self.table_name,
            'call_count': self.call_count,
            'total_time_ms': round(self.total_time_ms, 3),
            'avg_time_ms': round(self.avg_time_ms, 3),
            'min_time_ms': round(self.min_time_ms, 3) if self.min_time_ms != float('inf') else 0.0,
            'max_time_ms': round(self.max_time_ms, 3),
            'documents_scanned': self.documents_scanned,
            'documents_matched': self.documents_matched,
        }


@dataclass
class QuerySummary:
    """
    A summary of query profiling data.

    This class provides convenient access to the most important profiling
    insights, including the slowest queries and most frequently called queries.

    Attributes:
        total_queries: Total number of query executions
        total_time_ms: Total time spent executing queries in milliseconds
        unique_queries: Number of unique query types
        slowest_queries: List of queries sorted by average execution time
        most_called_queries: List of queries sorted by call count
        queries_by_table: Queries grouped by table name
    """
    total_queries: int = 0
    total_time_ms: float = 0.0
    unique_queries: int = 0
    slowest_queries: List[QueryStats] = field(default_factory=list)
    most_called_queries: List[QueryStats] = field(default_factory=list)
    queries_by_table: Dict[str, List[QueryStats]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy inspection."""
        return {
            'total_queries': self.total_queries,
            'total_time_ms': round(self.total_time_ms, 3),
            'unique_queries': self.unique_queries,
            'slowest_queries': [q.to_dict() for q in self.slowest_queries],
            'most_called_queries': [q.to_dict() for q in self.most_called_queries],
            'queries_by_table': {
                table: [q.to_dict() for q in queries]
                for table, queries in self.queries_by_table.items()
            },
        }

    def __repr__(self) -> str:
        lines = [
            '=' * 60,
            'QUERY PROFILING SUMMARY',
            '=' * 60,
            f'Total queries executed: {self.total_queries}',
            f'Total time: {self.total_time_ms:.3f} ms',
            f'Unique query types: {self.unique_queries}',
            '',
        ]

        if self.slowest_queries:
            lines.append('-' * 60)
            lines.append('SLOWEST QUERIES (by average time)')
            lines.append('-' * 60)
            for i, q in enumerate(self.slowest_queries[:10], 1):
                lines.append(
                    f'{i}. [{q.table_name}] {q.query_repr}'
                )
                lines.append(
                    f'   avg: {q.avg_time_ms:.3f}ms, '
                    f'calls: {q.call_count}, '
                    f'total: {q.total_time_ms:.3f}ms'
                )
            lines.append('')

        if self.most_called_queries:
            lines.append('-' * 60)
            lines.append('MOST FREQUENTLY CALLED QUERIES')
            lines.append('-' * 60)
            for i, q in enumerate(self.most_called_queries[:10], 1):
                lines.append(
                    f'{i}. [{q.table_name}] {q.query_repr}'
                )
                lines.append(
                    f'   calls: {q.call_count}, '
                    f'avg: {q.avg_time_ms:.3f}ms, '
                    f'total: {q.total_time_ms:.3f}ms'
                )
            lines.append('')

        if self.queries_by_table:
            lines.append('-' * 60)
            lines.append('QUERIES BY TABLE')
            lines.append('-' * 60)
            for table, queries in sorted(self.queries_by_table.items()):
                table_total_time = sum(q.total_time_ms for q in queries)
                table_total_calls = sum(q.call_count for q in queries)
                lines.append(
                    f'Table: {table} '
                    f'({len(queries)} unique queries, '
                    f'{table_total_calls} total calls, '
                    f'{table_total_time:.3f}ms total time)'
                )
            lines.append('')

        lines.append('=' * 60)
        return '\n'.join(lines)


def _format_query_hash(hashval: Optional[Tuple]) -> str:
    """
    Format a query hash tuple into a human-readable string.

    :param hashval: The hash tuple from a QueryInstance
    :return: A human-readable string representation
    """
    if hashval is None:
        return '<non-cacheable query>'

    if not hashval:
        return '<noop>'

    def format_value(val: Any) -> str:
        """Format a single value for display."""
        if isinstance(val, tuple):
            if len(val) == 2 and val[0] == 'path':
                # Path tuple like ('path', ('field', 'subfield'))
                return '.'.join(str(p) for p in val[1])
            return str(val)
        if isinstance(val, frozenset):
            # For AND/OR operations
            items = [format_value(v) for v in val]
            return ', '.join(sorted(items))
        if isinstance(val, str):
            return f"'{val}'"
        if callable(val):
            return f'<{val.__name__ if hasattr(val, "__name__") else "func"}>'
        return str(val)

    op = hashval[0]

    if op == 'path':
        # Simple field path like ('path', ('name',))
        return '.'.join(str(p) for p in hashval[1])

    if op in ('==', '!=', '<', '<=', '>', '>='):
        # Comparison operations like ('==', ('name',), 'John')
        path = '.'.join(str(p) for p in hashval[1])
        value = format_value(hashval[2])
        return f"{path} {op} {value}"

    if op in ('exists', 'matches', 'search'):
        # Existence and regex operations
        path = '.'.join(str(p) for p in hashval[1])
        if op == 'exists':
            return f"{path}.exists()"
        elif len(hashval) > 2:
            return f"{path}.{op}({format_value(hashval[2])})"
        return f"{path}.{op}()"

    if op in ('any', 'all', 'one_of'):
        # Collection operations
        path = '.'.join(str(p) for p in hashval[1])
        if len(hashval) > 2:
            return f"{path}.{op}({format_value(hashval[2])})"
        return f"{path}.{op}()"

    if op == 'test':
        # Custom test function
        path = '.'.join(str(p) for p in hashval[1])
        func_name = format_value(hashval[2]) if len(hashval) > 2 else '<func>'
        return f"{path}.test({func_name})"

    if op == 'fragment':
        # Fragment matching
        return f"fragment({format_value(hashval[1])})"

    if op == 'and':
        # AND operation
        parts = [format_value(v) for v in hashval[1]]
        return f"({' & '.join(sorted(parts))})"

    if op == 'or':
        # OR operation
        parts = [format_value(v) for v in hashval[1]]
        return f"({' | '.join(sorted(parts))})"

    if op == 'not':
        # NOT operation
        return f"~({format_value(hashval[1])})"

    # Fallback for unknown formats
    return str(hashval)


class QueryProfiler:
    """
    A profiler for tracking TinyDB query performance.

    The QueryProfiler collects timing and execution statistics for queries,
    allowing developers to identify slow queries and optimization opportunities.

    Usage:

    >>> from tinydb import TinyDB, where
    >>> from tinydb.profiling import QueryProfiler
    >>>
    >>> db = TinyDB('db.json')
    >>> profiler = QueryProfiler()
    >>> db.enable_profiling(profiler)
    >>>
    >>> # Perform queries
    >>> db.search(where('name') == 'John')
    >>> db.search(where('age') > 25)
    >>>
    >>> # View statistics
    >>> print(profiler.get_summary())
    >>>
    >>> # Get specific stats
    >>> for stats in profiler.get_slowest_queries(5):
    ...     print(f"{stats.query_repr}: {stats.avg_time_ms:.2f}ms")
    >>>
    >>> # Disable and clear
    >>> db.disable_profiling()
    >>> profiler.clear()
    """

    def __init__(self) -> None:
        """Initialize a new QueryProfiler."""
        # Key: (table_name, query_hash) -> QueryStats
        self._stats: Dict[Tuple[str, int], QueryStats] = {}
        # Store query representations for lookup
        self._query_reprs: Dict[Tuple[str, int], str] = {}
        self._enabled = True

    @property
    def enabled(self) -> bool:
        """Check if profiling is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable profiling."""
        self._enabled = value

    def record_query(
        self,
        table_name: str,
        query: Any,
        execution_time_ms: float,
        docs_scanned: int = 0,
        docs_matched: int = 0
    ) -> None:
        """
        Record a query execution.

        :param table_name: Name of the table the query was executed against
        :param query: The query object (must have __hash__ method)
        :param execution_time_ms: Execution time in milliseconds
        :param docs_scanned: Number of documents scanned
        :param docs_matched: Number of documents matched (must be <= docs_scanned)

        :raises ValueError: If docs_matched > docs_scanned, indicating a bug
                           in the profiling instrumentation
        """
        if not self._enabled:
            return

        # Validate that docs_matched cannot exceed docs_scanned
        # This would indicate a bug in the profiling instrumentation
        if docs_matched > docs_scanned:
            raise ValueError(
                f"docs_matched ({docs_matched}) cannot be greater than "
                f"docs_scanned ({docs_scanned}). This indicates a bug in "
                f"the profiling instrumentation for table '{table_name}'."
            )

        # Get query hash and representation
        query_hash = hash(query)
        key = (table_name, query_hash)

        # Generate query representation if not already cached
        if key not in self._query_reprs:
            hashval = getattr(query, '_hash', None)
            self._query_reprs[key] = _format_query_hash(hashval)

        # Get or create stats object
        if key not in self._stats:
            self._stats[key] = QueryStats(
                query_repr=self._query_reprs[key],
                table_name=table_name
            )

        # Record the execution
        self._stats[key].record(execution_time_ms, docs_scanned, docs_matched)

    def get_stats(self, table_name: Optional[str] = None) -> List[QueryStats]:
        """
        Get all recorded query statistics.

        :param table_name: Optional table name to filter by
        :return: List of QueryStats objects
        """
        if table_name is not None:
            return [
                stats for stats in self._stats.values()
                if stats.table_name == table_name
            ]
        return list(self._stats.values())

    def get_slowest_queries(
        self,
        limit: int = 10,
        table_name: Optional[str] = None
    ) -> List[QueryStats]:
        """
        Get the slowest queries by average execution time.

        :param limit: Maximum number of queries to return
        :param table_name: Optional table name to filter by
        :return: List of QueryStats sorted by average time (descending)
        """
        stats = self.get_stats(table_name)
        return sorted(stats, key=lambda s: s.avg_time_ms, reverse=True)[:limit]

    def get_most_called_queries(
        self,
        limit: int = 10,
        table_name: Optional[str] = None
    ) -> List[QueryStats]:
        """
        Get the most frequently called queries.

        :param limit: Maximum number of queries to return
        :param table_name: Optional table name to filter by
        :return: List of QueryStats sorted by call count (descending)
        """
        stats = self.get_stats(table_name)
        return sorted(stats, key=lambda s: s.call_count, reverse=True)[:limit]

    def get_queries_by_total_time(
        self,
        limit: int = 10,
        table_name: Optional[str] = None
    ) -> List[QueryStats]:
        """
        Get queries sorted by total execution time.

        This helps identify queries that may not be slow individually but
        consume significant time due to high call frequency.

        :param limit: Maximum number of queries to return
        :param table_name: Optional table name to filter by
        :return: List of QueryStats sorted by total time (descending)
        """
        stats = self.get_stats(table_name)
        return sorted(stats, key=lambda s: s.total_time_ms, reverse=True)[:limit]

    def get_summary(
        self,
        slowest_limit: int = 10,
        most_called_limit: int = 10
    ) -> QuerySummary:
        """
        Get a comprehensive summary of all profiling data.

        :param slowest_limit: Number of slowest queries to include
        :param most_called_limit: Number of most called queries to include
        :return: QuerySummary object with all statistics
        """
        all_stats = list(self._stats.values())

        # Group by table
        by_table: Dict[str, List[QueryStats]] = defaultdict(list)
        for stats in all_stats:
            by_table[stats.table_name].append(stats)

        return QuerySummary(
            total_queries=sum(s.call_count for s in all_stats),
            total_time_ms=sum(s.total_time_ms for s in all_stats),
            unique_queries=len(all_stats),
            slowest_queries=sorted(
                all_stats, key=lambda s: s.avg_time_ms, reverse=True
            )[:slowest_limit],
            most_called_queries=sorted(
                all_stats, key=lambda s: s.call_count, reverse=True
            )[:most_called_limit],
            queries_by_table=dict(by_table),
        )

    def clear(self) -> None:
        """Clear all recorded statistics."""
        self._stats.clear()
        self._query_reprs.clear()

    def __repr__(self) -> str:
        total_queries = sum(s.call_count for s in self._stats.values())
        unique_queries = len(self._stats)
        status = 'enabled' if self._enabled else 'disabled'
        return (
            f'<QueryProfiler {status}, '
            f'{unique_queries} unique queries, '
            f'{total_queries} total executions>'
        )
