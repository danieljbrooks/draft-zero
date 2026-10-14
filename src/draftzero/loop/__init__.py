"""The actor-learner loop (contracts §4, docs/architecture.md): roles that share nothing but a store.

    store       the shared store: LocalStore (a folder) and HFBucketStore (a Hugging Face Storage Bucket)
    control     the store's layout, control.json, plan requests and events, shard and version helpers
    actor       plays self-play chunks with the current version and uploads shards (many, any machine)
    learner     ingests and packs shards, trains on the newest window, publishes versions (one, holding a lease)
    evaluator   plays each new version against eval.opponents in paired games (a few)
    controller  the only writer of control.json: versions, plan changes, evaluation jobs, budget, status.md
    relay       moves shards and versions between a token-less machine's folder and the bucket, over ssh
    local       every role on one machine, as separate processes sharing a store
    cli         python -m draftzero.loop.cli <role>
"""
