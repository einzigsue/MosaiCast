"""Thin re-export / extension point for aurora.Batch.

aurora.Batch is used directly throughout; this module exists so downstream code
imports from mosaicast.batch (allowing future extensions without touching callers).
"""
from aurora import Batch, Metadata  # noqa: F401
