import flwr as fl
import ray
import os
import torch
from typing import List, Tuple, Dict, Optional, Callable
from flwr.common import Metrics, Parameters, Scalar
from client_app import create_client
from task import Net, get_num_classes

class PlottingStrategy(fl.server.strategy.FedAvg):
    def __init__(self, stats_callback: Callable, dataset_name: str, num_classes: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats_callback = stats_callback
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.accuracy_history = []
        self.loss_history = []
        self.logs = []

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
        
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        
        if examples and sum(examples) > 0:
            accuracy_aggregated = sum(accuracies) / sum(examples)
            
            self.accuracy_history.append(accuracy_aggregated)
            if loss_aggregated is not None:
                self.loss_history.append(loss_aggregated)
            
            # Update stats
            self.stats_callback(server_round, accuracy_aggregated, self.logs)
            
        return loss_aggregated, metrics_aggregated

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        
        if parameters_aggregated is not None:
            # Add logs
            self.logs.append(f"Round {server_round}: Received encrypted weights from {len(results)} clients.")
            self.logs.append(f"Round {server_round}: Aggregating model updates... No raw pixels shared.")
            
            # Simple callback update for logs even if accuracy hasn't changed yet (optional, or wait for eval)
            # We wait for eval to push everything usually, but let's push logs now
            # Only if we have history, otherwise 0
            curr_acc = self.accuracy_history[-1] if self.accuracy_history else 0.0
            self.stats_callback(server_round, curr_acc, self.logs)

            print(f"Saving global model round {server_round}...")
            # Initialize Net with correct dimensions for this dataset
            net = Net(num_classes=self.num_classes)
            params_dict = zip(net.state_dict().keys(), fl.common.parameters_to_ndarrays(parameters_aggregated))
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            net.load_state_dict(state_dict)
            
            # Save specific model file
            filename = f"global_model_{self.dataset_name}.pth"
            torch.save(net.state_dict(), filename)
            
        return parameters_aggregated, metrics_aggregated


def run_simulation(num_rounds: int, stats_callback: Callable, dataset_name: str = "chest_xray"):
    print(f"Starting simulation for dataset: {dataset_name}...")
    
    # 1. Determine num_classes for this dataset
    num_classes = get_num_classes(dataset_name)
    print(f"Detected {num_classes} classes for {dataset_name}")

    # 2. Configure Strategy with dataset info
    strategy = PlottingStrategy(
        stats_callback=stats_callback,
        dataset_name=dataset_name,
        num_classes=num_classes,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )

    # 3. Create client_fn wrapper to pass dataset_name
    def client_fn_wrapper(cid):
        return create_client(cid, dataset_name=dataset_name)

    # Initialize Ray with Windows-specific fix
    # Monkeypatch the function causing the Job Object error
    def noop(*args, **kwargs):
        pass
    
    try:
        import ray._private.utils
        ray._private.utils.set_kill_child_on_death_win32 = noop
    except ImportError:
        pass

    # shutdown() handles cases where a previous run left Ray in a bad state
    if ray.is_initialized():
        ray.shutdown()
        
    ray.init(
        runtime_env={"env_vars": {"RAY_CHROOT": "0"}},
        ignore_reinit_error=True,
        include_dashboard=False,
        local_mode=True, # Force single-process execution
        num_cpus=2 
    )

    try:
        fl.simulation.start_simulation(
            client_fn=client_fn_wrapper,
            num_clients=2,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0}
        )
    finally:
        # Clean up Ray resources to prevent Job Object conflicts on next run
        ray.shutdown()
