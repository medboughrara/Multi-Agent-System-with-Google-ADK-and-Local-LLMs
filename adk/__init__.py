"""
Google Agent Development Kit (ADK) for Local Models
This package provides tools for building multi-agent systems with local models.
"""

import logging
from typing import Any

from .components import Agent, Tool, Sequential, Parallel, Loop
from .runtime import Context

__version__ = "0.1.0"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
