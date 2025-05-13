"""
Runtime context management for ADK.
Provides tools for storing and retrieving data during workflow execution.
"""

from typing import Any, Dict, Set, Optional, List
import threading
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class Context:
    """
    Execution context for ADK workflows.
    Provides thread-safe storage and retrieval of data during workflow execution.
    """
    
    def __init__(self):
        """Initialize an empty context."""
        self._store: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._timestamps: Dict[str, datetime] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def store(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store a value in the context.
        
        Args:
            key: The key to store the value under
            value: The value to store
            metadata: Optional metadata about the stored value
        """
        with self._lock:
            self._store[key] = value
            self._timestamps[key] = datetime.now()
            if metadata:
                self._metadata[key] = metadata
    
    def retrieve(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from the context.
        
        Args:
            key: The key to retrieve
            default: Value to return if key doesn't exist
            
        Returns:
            The stored value or the default
        """
        with self._lock:
            return self._store.get(key, default)
    
    def store_keys(self) -> Set[str]:
        """
        Get all stored keys.
        
        Returns:
            Set of all keys in the store
        """
        with self._lock:
            return set(self._store.keys())
    
    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a stored value.
        
        Args:
            key: The key to get metadata for
            
        Returns:
            The metadata dictionary or None if not found
        """
        with self._lock:
            return self._metadata.get(key)
    
    def get_timestamp(self, key: str) -> Optional[datetime]:
        """
        Get the timestamp when a value was stored.
        
        Args:
            key: The key to get timestamp for
            
        Returns:
            Datetime when the value was stored or None if not found
        """
        with self._lock:
            return self._timestamps.get(key)
    
    def clear(self) -> None:
        """Clear all stored data."""
        with self._lock:
            self._store.clear()
            self._timestamps.clear()
            self._metadata.clear()
    
    def update(self, other: 'Context') -> None:
        """
        Update this context with values from another context.
        
        Args:
            other: Another Context instance to copy values from
        """
        with self._lock:
            self._store.update(other._store)
            self._timestamps.update(other._timestamps)
            self._metadata.update(other._metadata)
