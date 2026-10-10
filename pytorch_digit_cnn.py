
import torch, torch.nn as nn
from torchvision import datasets, transforms
torch.manual_seed(42)
tf = transforms.ToTensor()
tr = torch.utils.data.DataLoader(datasets.MNIST("data", True, download=True, transform=tf), batch_size=128, shuffle=True)
te = torch.utils.data.DataLoader(datasets.MNIST("data", False, download=True, transform=tf), batch_size=512)
net = nn.Sequential(nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Flatten(), nn.Linear(64*7*7,128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128,10))
opt = torch.optim.Adam(net.parameters(), 1e-3); loss_fn = nn.CrossEntropyLoss()
for ep in range(5):
    net.train()
    for x, y in tr:
        opt.zero_grad(); loss_fn(net(x), y).backward(); opt.step()
    net.eval(); ok = n = 0
    with torch.no_grad():
        for x, y in te: ok += (net(x).argmax(1) == y).sum().item(); n += len(y)
    print(f"epoch {ep+1}: test accuracy {ok/n:.4f}")
torch.save(net.state_dict(), "cnn.pt")
