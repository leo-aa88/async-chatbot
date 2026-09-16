"""Domain contracts: enums, events, state models, and LLM proposals.

Everything in this package is pure data (frozen dataclasses and enums) with no IO and no
behavior beyond validation. The reducer and stores own all persistence and mutation.
"""
