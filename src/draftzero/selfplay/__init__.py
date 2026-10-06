"""Self-play from the imitation start (docs/021): an actor-learner loop that runs on one machine or many.

Three roles, which share nothing but a store (a local folder, or Hugging Face):

    worker      plays self-play games with the current weights (PIMC search, the network on the machine's own
                GPU or CPU) and uploads them in batches; plays evaluation shards when the controller assigns them
    trainer     exactly one: reads new games, fine-tunes the network continuously (search targets, results, a KL
                penalty toward the starting network), publishes new versions
    controller  the only writer of control.json: which version to play, evaluation jobs and their sequential
                tests, a status page

    dz selfplay worker      --config configs/selfplay_gnn.yml --store <store>
    dz selfplay trainer     --config configs/selfplay_gnn.yml --store <store>
    dz selfplay controller  --config configs/selfplay_gnn.yml --store <store>
    dz selfplay local       --config configs/selfplay_smoke.yml --store runs/selfplay/smoke     all three on one box
    dz selfplay status      --store <store>

Modules:
    config      the run's settings (one YAML for every role)
    store       the shared store: LocalStore (a folder) and HFStore (a public dataset + a private weights repo)
    control     control.json's schema
    records     game batches: play.py's per-game records and results, packed for upload
    tables      batches -> training rows (graph rows for the GNN, soft tables for the flat MLP)
    gnn_train   the GNN's self-play training step (soft search targets, value, KL, human rows)
    worker, trainer, controller, evals, local, cli
"""
