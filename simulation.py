from logging import INFO
from typing import Any, Callable, Dict, List, Optional, Tuple

import ray
from flwr.client.client import Client
from flwr.common.logger import log
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg, Strategy
from flwr.simulation.ray_transport.ray_client_proxy import RayClientProxy

from server import Server, save_metrics, save_model


def start_simulation(  # pylint: disable=too-many-arguments
        *,
        client_fn: Callable[[str], Client],
        num_clients: Optional[int] = None,
        client_resources: Optional[Dict[str, int]] = None,
        num_rounds: int = 1,
        strategy: Optional[Strategy] = None,
        ray_init_args: Optional[Dict[str, Any]] = None,
        path_to_save_metrics,
        model,
) -> None:
    """Start a Ray-based Flower simulation server.
    Parameters
    ----------
    client_fn : Callable[[str], Client]
        A function creating client instances. The function must take a single
        str argument called `cid`. It should return a single client instance.
        Note that the created client instances are ephemeral and will often be
        destroyed after a single method invocation. Since client instances are
        not long-lived, they should not attempt to carry state over method
        invocations. Any state required by the instance (model, dataset,
        hyperparameters, ...) should be (re-)created in either the call to
        `client_fn` or the call to any of the client methods (e.g., load
        evaluation data in the `evaluate` method itself).
    num_clients : Optional[int]
        The total number of clients in this simulation. This must be set if
        `clients_ids` is not set and vice-versa.
    clients_ids : Optional[List[str]]
        List `client_id`s for each client. This is only required if
        `num_clients` is not set. Setting both `num_clients` and `clients_ids`
        with `len(clients_ids)` not equal to `num_clients` generates an error.
    client_resources : Optional[Dict[str, int]] (default: None)
        CPU and GPU resources for a single client. Supported keys are
        `num_cpus` and `num_gpus`. Example: `{"num_cpus": 4, "num_gpus": 1}`.
        To understand the GPU utilization caused by `num_gpus`, consult the Ray
        documentation on GPU support.
    num_rounds : int (default: 1)
        The number of rounds to train.
    strategy : Optional[flwr.server.Strategy] (default: None)
        An implementation of the abstract base class `flwr.server.Strategy`. If
        no strategy is provided, then `start_server` will use
        `flwr.server.strategy.FedAvg`.
    ray_init_args : Optional[Dict[str, Any]] (default: None)
        Optional dictionary containing arguments for the call to `ray.init`.
        If ray_init_args is None (the default), Ray will be initialized with
        the following default args:
            {
                "ignore_reinit_error": True,
                "include_dashboard": False,
            }
        An empty dictionary can be used (ray_init_args={}) to prevent any
        arguments from being passed to ray.init.
    """
    cids: List[str] = [str(x) for x in range(num_clients)]

    # Default arguments for Ray initialization
    if not ray_init_args:
        ray_init_args = {
            "ignore_reinit_error": True,
            "include_dashboard": False,
        }

    # Shut down Ray if it has already been initialized
    if ray.is_initialized():
        ray.shutdown()

    # Initialize Ray
    ray.init(**ray_init_args)
    log(
        INFO,
        "Ray initialized with resources: %s",
        ray.cluster_resources(),
    )

    # Initialize server and server config
    config = {"num_rounds": num_rounds, "path_to_save_metrics": path_to_save_metrics}
    initialized_server, initialized_config = _init_defaults(None, config, strategy, model)
    log(
        INFO,
        "Starting Flower simulation running: %s",
        initialized_config,
    )

    # Register one RayClientProxy object for each client with the ClientManager
    resources = client_resources if client_resources is not None else {}
    for cid in cids:
        client_proxy = RayClientProxy(
            client_fn=client_fn,
            cid=cid,
            resources=resources,
        )
        initialized_server.client_manager().register(client=client_proxy)

    # Start training
    _fl(
        server=initialized_server,
        config=initialized_config,
        model=model
    )


def _init_defaults(
        server: Optional[Server],
        config: Optional[Dict[str, int]],
        strategy: Optional[Strategy],
        model
) -> Tuple[Server, Dict[str, int]]:
    # Create server instance if none was given
    if server is None:
        client_manager = SimpleClientManager()
        if strategy is None:
            strategy = FedAvg()
        server = Server(client_manager=client_manager, strategy=strategy,
                        path_to_save_metrics=config["path_to_save_metrics"], model=model)

    # Set default config values
    if config is None:
        config = {}
    if "num_rounds" not in config:
        config["num_rounds"] = 1

    return server, config


def _fl(
        server: Server, config: Dict[str, int], model
) -> None:
    # Fit model
    hist = server.fit(num_rounds=config["num_rounds"])
    save_metrics(hist, config["path_to_save_metrics"])
    save_model(server.parameters, model, config["path_to_save_metrics"])

    log(INFO, "app_fit: losses_distributed %s", str(hist.losses_distributed))
    log(INFO, "app_fit: metrics_distributed %s", str(hist.metrics_distributed))
    log(INFO, "app_fit: losses_centralized %s", str(hist.losses_centralized))
    log(INFO, "app_fit: metrics_centralized %s", str(hist.metrics_centralized))

    # Graceful shutdown
    server.disconnect_all_clients()
