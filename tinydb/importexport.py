"""
This module provides import and export functionality for TinyDB tables.

It supports exporting table data to CSV and JSONL formats, as well as
importing data from those formats back into tables. Document IDs are
preserved during export/import to ensure data consistency.
"""

import csv
import json
from pathlib import Path
from typing import (
    Any,
    Dict,
    IO,
    List,
    Mapping,
    Optional,
    Union,
)

__all__ = (
    'export_csv',
    'import_csv',
    'export_jsonl',
    'import_jsonl',
)

#: The key used to store document ID in exported data
DOC_ID_KEY = '_id'


def export_csv(
    table: 'Table',
    file_path: Union[str, Path],
    *,
    include_deleted: bool = False,
    encoding: str = 'utf-8',
) -> int:
    """
    Export table data to a CSV file.

    Document IDs are preserved as the '_id' column. Complex values (dicts,
    lists) are serialized as JSON strings.

    :param table: The table to export from
    :param file_path: Path to the output CSV file
    :param include_deleted: If True, include soft-deleted documents.
                           Default is False.
    :param encoding: File encoding (default: 'utf-8')
    :returns: Number of documents exported

    Example usage:

    >>> from tinydb import TinyDB
    >>> from tinydb.importexport import export_csv
    >>> db = TinyDB('db.json')
    >>> table = db.table('users')
    >>> count = export_csv(table, 'users_backup.csv')
    >>> print(f"Exported {count} documents")
    """
    docs = table.all(include_deleted=include_deleted)

    if not docs:
        # Write empty file with no headers
        with open(file_path, 'w', newline='', encoding=encoding) as f:
            pass
        return 0

    # Collect all unique field names across all documents
    fieldnames = _collect_fieldnames(docs)

    with open(file_path, 'w', newline='', encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for doc in docs:
            row = _document_to_csv_row(doc)
            writer.writerow(row)

    return len(docs)


def import_csv(
    table: 'Table',
    file_path: Union[str, Path],
    *,
    encoding: str = 'utf-8',
) -> List[int]:
    """
    Import data from a CSV file into a table.

    If the CSV has an '_id' column, document IDs are preserved. Complex
    values (dicts, lists) stored as JSON strings are automatically
    deserialized.

    :param table: The table to import into
    :param file_path: Path to the input CSV file
    :param encoding: File encoding (default: 'utf-8')
    :returns: List of imported document IDs

    Example usage:

    >>> from tinydb import TinyDB
    >>> from tinydb.importexport import import_csv
    >>> db = TinyDB('db.json')
    >>> table = db.table('users')
    >>> doc_ids = import_csv(table, 'users_backup.csv')
    >>> print(f"Imported {len(doc_ids)} documents")
    """
    # Import Document here to avoid circular imports
    from .table import Document

    doc_ids: List[int] = []

    with open(file_path, 'r', newline='', encoding=encoding) as f:
        reader = csv.DictReader(f)

        for row in reader:
            doc_data, doc_id = _csv_row_to_document(row, table.document_id_class)

            if doc_id is not None:
                # Create Document with preserved ID
                document = Document(doc_data, doc_id)
                inserted_id = table.insert(document)
            else:
                # Let table generate the ID
                inserted_id = table.insert(doc_data)

            doc_ids.append(inserted_id)

    return doc_ids


def export_jsonl(
    table: 'Table',
    file_path: Union[str, Path],
    *,
    include_deleted: bool = False,
    encoding: str = 'utf-8',
) -> int:
    """
    Export table data to a JSONL (JSON Lines) file.

    Each document is written as a single JSON object per line. Document
    IDs are preserved as the '_id' field.

    :param table: The table to export from
    :param file_path: Path to the output JSONL file
    :param include_deleted: If True, include soft-deleted documents.
                           Default is False.
    :param encoding: File encoding (default: 'utf-8')
    :returns: Number of documents exported

    Example usage:

    >>> from tinydb import TinyDB
    >>> from tinydb.importexport import export_jsonl
    >>> db = TinyDB('db.json')
    >>> table = db.table('users')
    >>> count = export_jsonl(table, 'users_backup.jsonl')
    >>> print(f"Exported {count} documents")
    """
    docs = table.all(include_deleted=include_deleted)

    with open(file_path, 'w', encoding=encoding) as f:
        for doc in docs:
            row = {DOC_ID_KEY: doc.doc_id}
            row.update(doc)
            # Use ensure_ascii=False to write actual UTF-8 characters
            # instead of \uXXXX escape sequences
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    return len(docs)


def import_jsonl(
    table: 'Table',
    file_path: Union[str, Path],
    *,
    encoding: str = 'utf-8',
) -> List[int]:
    """
    Import data from a JSONL (JSON Lines) file into a table.

    If documents have an '_id' field, document IDs are preserved.

    :param table: The table to import into
    :param file_path: Path to the input JSONL file
    :param encoding: File encoding (default: 'utf-8')
    :returns: List of imported document IDs

    Example usage:

    >>> from tinydb import TinyDB
    >>> from tinydb.importexport import import_jsonl
    >>> db = TinyDB('db.json')
    >>> table = db.table('users')
    >>> doc_ids = import_jsonl(table, 'users_backup.jsonl')
    >>> print(f"Imported {len(doc_ids)} documents")
    """
    # Import Document here to avoid circular imports
    from .table import Document

    doc_ids: List[int] = []

    with open(file_path, 'r', encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            doc_id = data.pop(DOC_ID_KEY, None)

            if doc_id is not None:
                # Convert to document_id_class
                doc_id = table.document_id_class(doc_id)
                # Create Document with preserved ID
                document = Document(data, doc_id)
                inserted_id = table.insert(document)
            else:
                # Let table generate the ID
                inserted_id = table.insert(data)

            doc_ids.append(inserted_id)

    return doc_ids


def _collect_fieldnames(docs: List['Document']) -> List[str]:
    """
    Collect all unique field names from documents, with _id first.

    :param docs: List of documents
    :returns: List of field names with _id first
    """
    fieldnames_set = set()
    for doc in docs:
        fieldnames_set.update(doc.keys())

    # Sort for consistent output, with _id first
    fieldnames = sorted(fieldnames_set)
    fieldnames.insert(0, DOC_ID_KEY)

    return fieldnames


def _document_to_csv_row(doc: 'Document') -> Dict[str, str]:
    """
    Convert a document to a CSV row dictionary.

    Complex values (dicts, lists) are serialized as JSON strings.

    :param doc: The document to convert
    :returns: Dictionary suitable for csv.DictWriter
    """
    row: Dict[str, str] = {DOC_ID_KEY: str(doc.doc_id)}

    for key, value in doc.items():
        if isinstance(value, (dict, list)):
            # Serialize complex types as JSON
            row[key] = json.dumps(value)
        elif value is None:
            row[key] = ''
        elif isinstance(value, bool):
            # Preserve boolean type with special marker
            row[key] = json.dumps(value)
        else:
            row[key] = str(value)

    return row


def _csv_row_to_document(
    row: Dict[str, str],
    document_id_class: type,
) -> tuple:
    """
    Convert a CSV row to document data and optional doc_id.

    Attempts to deserialize JSON strings back to complex types and
    converts numeric strings back to numbers.

    :param row: The CSV row dictionary
    :param document_id_class: The class to use for document IDs
    :returns: Tuple of (document_data dict, doc_id or None)
    """
    doc_data: Dict[str, Any] = {}
    doc_id: Optional[int] = None

    for key, value in row.items():
        if key == DOC_ID_KEY:
            if value:
                doc_id = document_id_class(value)
            continue

        # Skip empty values (they represent None)
        if value == '':
            doc_data[key] = None
            continue

        # Try to parse as JSON first (handles dicts, lists, booleans)
        try:
            parsed = json.loads(value)
            doc_data[key] = parsed
            continue
        except (json.JSONDecodeError, ValueError):
            pass

        # Try to convert to int
        try:
            doc_data[key] = int(value)
            continue
        except ValueError:
            pass

        # Try to convert to float
        try:
            doc_data[key] = float(value)
            continue
        except ValueError:
            pass

        # Keep as string
        doc_data[key] = value

    return doc_data, doc_id


# Type hint for Table (avoiding circular import)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .table import Table, Document
