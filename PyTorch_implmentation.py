import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

import run_ as data_gen  # Importiere das ursprüngliche Datengenerierungs-Skript


# Datensätze erstellen
def generate_data(num_samples=1000):
    # Importiere Rohdaten (Listen von numpy arrays) aus run_.py
    images, labels = data_gen.generate_data(num_samples)
    
    # Konvertierung zu numpy array (float32 für PyTorch)
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
    # WICHTIG: Normalisierung (Zentrierung um 0) nachholen! -> [0,1] zu [-0.5,0.5]
    # Im Originalskript (run_.py) passiert dies erst im forward-Pass.
    # Hier machen wir es direkt vor der Tensor-Erstellung.
    images -= 0.5 
    
    # Konvertierung zu PyTorch Tensoren
    # Form: (N, Channel, Height, Width) -> (N, 1, 8, 8)
    X = torch.tensor(images).unsqueeze(1)
    y = torch.tensor(labels)
    return X, y

train_X, train_y = generate_data(1000)
test_X, test_y = generate_data(200)


# Der DataLoader: Ersetzt unsere manuellen Batches und Shuffle-Logik
train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=32, shuffle=True)


# 2. Die Architektur (Das nn.Module)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Layer 1: Conv (Ersetzt unsere Conv3x3 Klasse)
        # in_channels=1 (Graustufen), out=8 (Filter), kernel=3
        self.conv = nn.Conv2d(1, 8, kernel_size=3) 
        
        # Layer 2: MaxPool (Ersetzt unsere MaxPool2 Klasse)
        self.pool = nn.MaxPool2d(2)
        
        # Layer 3: Linear (Ersetzt unsere Softmax Klasse)
        # 8x8 -> Conv(valid) -> 6x6 -> Pool -> 3x3.
        # Flatten Input = 8 Filter * 3 * 3 = 72
        self.fc = nn.Linear(8 * 3 * 3, 2) 

    def forward(self, x):
        # Hier definieren wir den Datenfluss
        x = self.conv(x)
        x = torch.relu(x)      # Die Aktivierung (implizit)
        x = self.pool(x)
        
        # Flatten: (Batch_Size, 8, 3, 3) -> (Batch_Size, 72)
        x = x.view(x.size(0), -1) 
        
        # Klassifikation (Raw Logits, kein Softmax hier, s.u.)
        x = self.fc(x)
        return x

# Initialisierung
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Gentztes Device: {device}")

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss() # Kombiniert Softmax + NLLLoss (Numerisch stabil!)
# Adam Optimizer (Adaptive Moment Estimation) etwas komplexer als grad_neu = lr * dL/dW 
# -> Gewichtete nach adaptiven Lernraten: 
# Gedächtnis (Momentum). Wenn er sich schon lange in eine Richtung bewegt, behält er den Schwung bei
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 3. Training Loop
print("Starte PyTorch Training...")
loss_history = []

for epoch in range(5):
    epoch_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # A. Reset Gradienten (Wichtig!)
        optimizer.zero_grad()
        
        # B. Forward Pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # C. Backward Pass (Die ganze Magie passiert hier)
        loss.backward() # Berechnet dL/dW für alle Layer automatisch
        
        # D. Update
        optimizer.step() # weights -= lr * gradient
        
        # Tracking
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = epoch_loss / len(train_loader)
    acc = 100 * correct / total
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}: Loss {avg_loss:.4f} | Accuracy {acc:.2f}%")

# 4. Analyse der Filter und plotte (Vergleich mit NumPy)
print("\nExtrahiere gelernte Kernel...")
filters = model.conv.weight.data.cpu().numpy() # Zugriff auf die Rohdaten (zurück auf CPU für NumPy)

fig, axes = plt.subplots(1, 8, figsize=(15, 3))
fig.suptitle('PyTorch: Gelernte 3x3 Filter', fontsize=16)
for i, ax in enumerate(axes):
    ax.imshow(filters[i, 0, :, :], cmap='gray', interpolation='nearest')
    ax.axis('off')
    ax.set_title(f'Filter {i+1}')

print("Loss History:", loss_history)
plt.show()
# plt.savefig('pytorch_kernels.png')
# print("Kernels gespeichert als 'pytorch_kernels.png'")