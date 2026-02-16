import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import argparse
from prometheus_client import start_http_server, Gauge
import time
import os

def save_checkpoint(state, filename="checkpoints/checkpoint.pth.tar"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)

def load_checkpoint(filename="checkpoints/checkpoint.pth.tar"):
    if os.path.isfile(filename):
        print(f"=> loading checkpoint '{filename}'")
        checkpoint = torch.load(filename, map_location='cpu')
        return checkpoint
    return None

def main():
    parser = argparse.ArgumentParser(description="Distributed Training Simulator")
    parser.add_argument('--world-size', type=int, default=int(os.environ.get('WORLD_SIZE', 1)), help='Total number of processes')
    parser.add_argument('--rank', type=int, default=int(os.environ.get('RANK', 0)), help='Rank of this process')
    parser.add_argument('--local-rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)), help='Local rank for GPU (if any)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()

    # Initialize distributed environment
    dist.init_process_group(backend='gloo', init_method='env://', world_size=args.world_size, rank=args.rank)

    # Prometheus Setup
    start_http_server(8000 + args.rank)
    loss_gauge = Gauge('training_loss', 'Current training loss', ['rank'])
    accuracy_gauge = Gauge('training_accuracy', 'Current training accuracy', ['rank'])
    throughput_gauge = Gauge('samples_per_second', 'Training throughput', ['rank'])

    device = torch.device('cpu')

    # Model
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(-1, 64 * 7 * 7)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = SimpleCNN().to(device)
    model = DDP(model, device_ids=[args.local_rank] if device.type == 'cuda' else None)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Fault Tolerance: Load Checkpoint
    start_epoch = 0
    checkpoint = load_checkpoint()
    if checkpoint:
        model.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        if args.rank == 0:
            print(f"=> resumed from epoch {start_epoch}")

    # Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler)

    # Training Loop
    model.train()
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = 100. * correct / total
        elapsed = time.time() - start_time
        throughput = total / elapsed

        loss_gauge.labels(rank=args.rank).set(avg_loss)
        accuracy_gauge.labels(rank=args.rank).set(accuracy)
        throughput_gauge.labels(rank=args.rank).set(throughput)

        if args.rank == 0:
            print(f"Epoch {epoch+1}: Loss {avg_loss:.4f}, Accuracy {accuracy:.2f}%, Throughput {throughput:.2f} samples/sec")
            # Fault Tolerance: Save Checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            })

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
