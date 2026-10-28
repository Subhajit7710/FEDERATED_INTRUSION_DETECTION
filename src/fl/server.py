
# File: src/fl/server.py

import flwr as fl


# --------------------------------------------------
# Federated Learning Configuration
# --------------------------------------------------

NUM_ROUNDS = 1

SERVER_ADDRESS = "0.0.0.0:8081"


# --------------------------------------------------
# Configuration sent to clients
# --------------------------------------------------

def fit_config(server_round):

    return {
        "local_epochs": 1
    }


# --------------------------------------------------
# FedAvg Strategy
# --------------------------------------------------

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,

    min_fit_clients=4,
    min_evaluate_clients=4,
    min_available_clients=4,

    on_fit_config_fn=fit_config
)


# --------------------------------------------------
# Start Flower Server
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 50)
    print("Starting Flower Federated Learning Server")
    print("=" * 50)

    print(f"Server address: {SERVER_ADDRESS}")
    print(f"Number of rounds: {NUM_ROUNDS}")
    print("Number of clients: 4")
    print("Aggregation: FedAvg")
    print("Local epochs: 1")

    print("=" * 50)

    fl.server.start_server(
        server_address=SERVER_ADDRESS,

        config=fl.server.ServerConfig(
            num_rounds=NUM_ROUNDS
        ),

        strategy=strategy
    )
