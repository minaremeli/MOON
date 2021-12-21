import flwr as fl
from flwr.common.typing import Scalar
import ray
import argparse
import torch
import numpy as np
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Callable, Optional, Tuple
from models import Cifar10Net, Cifar100Net
from strategy import FedAvg
from simulation import start_simulation
from dataset_utils import get_dataloader, do_fl_partitioning, getCIFAR100, getCIFAR10
import copy



def calc_loss(local_net, previous_net, global_net, x, target, mu, temperature, device):
    cos=torch.nn.CosineSimilarity(dim=-1)
    criterion = torch.nn.CrossEntropyLoss()
    pro1, out = local_net(x)
    pro2, _ = global_net(x)
    pro3, _ = previous_net(x)

    posi = cos(pro1, pro2)
    nega = cos(pro1, pro3)

    logits = posi.reshape(-1,1)
    logits = torch.cat((logits, nega.reshape(-1,1)), dim=1).to(device)
    logits /= temperature

    loss1 = criterion(out, target)
    t = torch.zeros(x.size(0)).to(device).long()
    loss2 = mu * criterion(logits, t)
    loss = loss1 + loss2
    
    return loss, out

def train(local_net, previous_net, global_net, trainloader, epochs, mu, temperature, device: str):
    local_net.train()
    optimizer = torch.optim.SGD(local_net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.00001)
    for _ in range(epochs):
        for (x, target) in trainloader:
            optimizer.zero_grad()

            x, target = x.to(device), target.to(device)
            loss, _ = calc_loss(local_net, previous_net, global_net, x, target, mu, temperature, device)
            
            loss.backward()
            optimizer.step()


# borrowed from Pytorch quickstart example
def test(net, testloader, device: str):
    """Validate the network on the entire test set."""
    correct, total = 0, 0
    net.eval()
    with torch.no_grad():
        for x,target in testloader:
            x, target = x.to(device), target.to(device)
            _, out = net(x)
            _, predicted = torch.max(out.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    return accuracy

def configure_net_for_eval(net):
    net.eval()
    for param in net.parameters():
        param.requires_grad = False

# Flower client that will be spawned by Ray
# Adapted from Pytorch quickstart example
class CifarRayClient(fl.client.NumPyClient):
    def __init__(self, dataset_name, cid: str, fed_dir: str):
        self.cid = cid
        self.fed_dir= Path(fed_dir)
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.temperature = 0.5
        self.dataset_name = dataset_name
        if dataset_name == 'CIFAR-10':
            self.mu = 5
            self.net = Cifar10Net()
        else: 
            self.mu = 1
            self.net = Cifar100Net()
        # determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    # def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
    def get_properties(self, ins):
        return self.properties

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        )
        self.net.load_state_dict(state_dict, strict=True)
    def load_prev_net(self):
        path_to_prev_net = self.fed_dir / self.cid / "net.pt"
        prev_net = copy.deepcopy(self.net)
        prev_net.load_state_dict(torch.load(path_to_prev_net))
        configure_net_for_eval(prev_net)
        return prev_net

    def fit(self, parameters, config):
        
        # print(f"fit() on client cid={self.cid}")
        self.set_parameters(parameters)
    
        # load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        trainloader = get_dataloader(
            self.dataset_name,
            self.fed_dir,
            self.cid,
            is_train=True,
            batch_size=int(config["batch_size"]),
            workers=num_workers,
        )
        # send model to device
        self.net.to(self.device)
        
        global_net = copy.deepcopy(self.net)
        configure_net_for_eval(global_net)
        global_net.to(self.device)

        prev_net = self.load_prev_net()
        prev_net.to(self.device)
        train(self.net, prev_net, global_net, trainloader, epochs=int(config["epochs"]), mu=self.mu, temperature=self.temperature, device=self.device)
        torch.save(self.net.state_dict(), self.fed_dir / str(self.cid) / "net.pt")
        # return local model and statistics
        return self.get_parameters(), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):

        # print(f"evaluate() on client cid={self.cid}")
        self.set_parameters(parameters)

        # load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        testloader = get_dataloader(
            self.dataset_name, self.fed_dir, self.cid, is_train=False, batch_size=60, workers=num_workers
        )
        # send model to device
        self.net.to(self.device)

        # evaluate
        accuracy = test(self.net, testloader, device=self.device)
        # return statistics
        return 1.0, len(testloader.dataset), {"accuracy": float(accuracy)}
def get_eval_fn(model, testset) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:


        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        set_weights(model, weights)
        model.to(device)

        testloader = torch.utils.data.DataLoader(testset, batch_size=50)
        accuracy = test(model, testloader, device=device)

        # return statistics
        return 0, {"accuracy": accuracy}

    return evaluate

def set_weights(model: torch.nn.ModuleList, weights: fl.common.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)

def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(rnd),
        "epochs": str(10),
        "batch_size": str(64),
    }
    return config
def main():
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=3,
        help="Number of rounds of federated learning (default: 1)",
    )
    parser.add_argument(
        "--sample_fraction_fit",
        type=float,
        default=0.1,
        help="Fraction of available clients used for fit (default: 1.0)",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=100,
        help="Number of clients to start",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="fedAvg",
        help="Name of the strategy you are running (default: fedAvg)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='CIFAR-10',
        help="Dataset to train on. Currently support only CIFAR-10 and CIFAR-100."
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1000,
        help="Concentration parameter of Dirichlet Distribution."
    )

    args = parser.parse_args()
    pool_size = args.num_clients  # number of dataset partions (= number of total clients)
    num_classes = 10 if args.dataset=='CIFAR-10' else 100
    model = Cifar10Net() if args.dataset == "CIFAR-10" else Cifar100Net()

    # download CIFAR10 dataset
    train_path, test_path, test_set = getCIFAR10() if args.dataset=='CIFAR-10' else getCIFAR100()

    alpha = args.alpha
    

    # partition dataset (use a large `alpha` to make it IID;
    # a small value (e.g. 1) will make it non-IID)
    # This will create a new directory called "federated: in the directory where
    # CIFAR-10 lives. Inside it, there will be N=pool_size sub-directories each with
    # its own train/set split.
    dirichlet_dist = np.random.default_rng().dirichlet(alpha=np.repeat(alpha, num_classes), size=pool_size)
    fed_dir= do_fl_partitioning(
        train_path, pool_size=pool_size, is_train=True, dirichlet_dist=dirichlet_dist
    )
    do_fl_partitioning(
        test_path, pool_size=pool_size, is_train=False, dirichlet_dist=dirichlet_dist
    )
    prev_net = Cifar10Net() if args.dataset == 'CIFAR-10' else Cifar100Net()
    for i in range(pool_size):
        torch.save(prev_net.state_dict(), fed_dir / str(i) / "net.pt")
    
    # configure the strategy
  
    strategy = FedAvg(
        fraction_fit=args.sample_fraction_fit,
        min_available_clients=pool_size,  # All clients should be available
        on_fit_config_fn=fit_config,
        eval_fn=get_eval_fn(model, test_set),  # centralised testset evaluation of global model
    )


    def client_fn(cid: str):
        # create a single client instance
        return CifarRayClient(args.dataset, cid, fed_dir)

    # (optional) specify ray config
    ray_config = {"include_dashboard": False}

    # start simulation
    start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        num_rounds=args.num_rounds,
        strategy=strategy,
        model = model,
        ray_init_args=ray_config,
        path_to_save_metrics = fed_dir.parent.parent
    )

if __name__ == "__main__":
    main()
    