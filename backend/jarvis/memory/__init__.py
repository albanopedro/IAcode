"""JARVIS memory: persistent history, conversation summaries and long-term facts.

Everything stays in a local SQLite file (``data/memory.db``). Nothing is sent
anywhere except, as context, to the agents that answer you.
"""
