from collections import OrderedDict
import torch
import flwr as fl
from task import Net, load_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, num_classes):
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = Net(num_classes=num_classes).to(DEVICE)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train()
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.test()
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}

    def train(self):
        """Train the model on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.model.train()
        for epoch in range(1): # Train for 1 epoch per round
            for images, labels in self.trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(self.model(images), labels)
                loss.backward()
                optimizer.step()

    def test(self):
        """Validate the model on the validation set."""
        criterion = torch.nn.CrossEntropyLoss()
        self.model.eval()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in self.valloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return loss / len(self.valloader), correct / total

def create_client(cid, dataset_name="chest_xray"):
    """Create a Flower client for the given partition and dataset."""
    # Handle cid being context or string depending on Flower version/setup
    if hasattr(cid, "node_config"):
        partition_id = int(cid.node_config["partition-id"])
    else:
        partition_id = int(cid)
        
    num_partitions = 2 # Fixed for this example
    
    # Load data for specific dataset
    trainloader, valloader, num_classes = load_data(
        dataset_name=dataset_name, 
        partition_id=partition_id, 
        num_partitions=num_partitions
    )
    
    return FlowerClient(trainloader, valloader, num_classes).to_client()

# Legacy adapter if needed directly
def client_fn(cid):
    return create_client(cid, "chest_xray")
