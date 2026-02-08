import flwr as fl
import ray
import os
from client_app import client_fn

def main():
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0, # Sample 100% of available clients for training
        fraction_evaluate=1.0, # Sample 100% of available clients for evaluation
        min_fit_clients=2, # Never train on fewer than 2 clients
        min_evaluate_clients=2, # Never evaluate on fewer than 2 clients
        min_available_clients=2, # Wait until all 2 clients are available
    )

    # Initialize Ray with Windows-specific fix
    if ray.is_initialized():
        ray.shutdown()

    ray.init(
        runtime_env={"env_vars": {"RAY_CHROOT": "0"}},
        ignore_reinit_error=True,
        include_dashboard=False
    )

    try:
        # Start simulation
        fl.simulation.run_simulation(
            client_fn=client_fn,
            num_clients=2,
            config=fl.server.ServerConfig(num_rounds=3),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0} # Adjust if you have a GPU
        )
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()
