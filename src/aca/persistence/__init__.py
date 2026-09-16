"""Durable persistence (SQLite).

This package is the only place that performs durable-state IO. The reducer is the single
writer; stores expose typed CRUD and never contain policy. Durable persistence is the
continuity boundary of the logical agent (DESIGN 23).
"""
