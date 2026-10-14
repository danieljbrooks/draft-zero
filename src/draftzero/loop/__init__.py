"""The actor-learner loop: roles that share nothing but a store (a local folder, or Hugging Face).

v0.2.0 is rewriting this package around gorge (docs/033): the actor and learner replace the XMage-era worker and
trainer, which are gone. What is here so far carries over from docs/021:

    store       the shared store: LocalStore (a folder) and HFStore (Hugging Face)
    control     control.json's schema
    controller  the only writer of control.json: versions, evaluation jobs and their sequential tests, status.md
    evals       evaluation jobs, shards, scoring and the sequential test
    settings    the docs/021 run settings (one YAML for every role); draftzero.config replaces it
    local       every role on one machine, as separate processes sharing a store
    cli         python -m draftzero.loop.cli <role>
"""
