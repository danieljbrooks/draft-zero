"""DraftZero: agents that play Magic: The Gathering limited tournaments end to end.

Games run in gorge (the pinned engine fork, engine/gorge). DraftZero owns the format-level concerns: the networks
and their training loop (nets, loop), deck pools (data), deck building (build), evaluation (eval) and the machines
that run it all (workers).
"""
__version__ = "0.1.0"
