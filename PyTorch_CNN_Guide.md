# PyTorch CNN - Basis-Vorlage & Leitfaden

Diese Anleitung dient als "Boilerplate" (Grundgerüst) für den Aufbau eines Convolutional Neural Networks (CNN) in PyTorch. Sie basiert auf gängigen Best Practices und der Syntax, die auch im Projekt `CNN_from_Scratch` verwendet wurde.

---

## 1. Imports & Setup

Hier werden die notwendigen Module geladen. `torch.nn` enthält die Layer, `torch.optim` die Optimierungsalgorithmen.

```python
import torch
import torch.nn as nn                  # Neural Network Module (Layer, Loss)
import torch.optim as optim            # Optimizer (SGD, Adam, etc.)
from torch.utils.data import DataLoader, TensorDataset # Daten-Management
import numpy as np
```

## 2. Hardware-Beschleunigung (CUDA/MPS)

PyTorch wählt nicht automatisch die GPU. Wir müssen das `device` explizit definieren.

```python
# Prüft, ob CUDA (Nvidia) verfügbar ist, sonst CPU. 
# (Auf Macs mit M1/M2 Chips wäre dies "mps", hier allgemein gehalten für Linux/Windows)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Genutztes Device: {device}")
```

## 3. Datenvorbereitung (Formatierung)

PyTorch CNNs erwarten Input-Daten strikt im Format: **`(N, C, H, W)`**.
*   **N**: Batch Size (Anzahl der Bilder)
*   **C**: Channels (Kanäle, z.B. 1 für Graustufen, 3 für RGB)
*   **H**: Höhe
*   **W**: Breite

```python
# Beispielhafte Konvertierung von NumPy Arrays zu PyTorch Tensoren
# Angenommen 'images' ist ein numpy array (Anzahl, Höhe, Breite)
def prepare_data(images, labels):
    # 1. Normalisierung: Wichtig für Konvergenz (oft auf [-0.5, 0.5] oder [0, 1])
    images = images / 255.0 - 0.5 
    
    # 2. Zu Tensor konvertieren (Float für Input, Long für Klassen-Labels)
    x_tensor = torch.tensor(images, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    
    # 3. Channel-Dimension hinzufügen, falls sie fehlt (z.B. bei Graustufen)
    # Von (N, H, W) -> (N, 1, H, W)
    if len(x_tensor.shape) == 3:
        x_tensor = x_tensor.unsqueeze(1)
        
    return x_tensor, y_tensor

# DataLoader erstellt Batches und shuffelt die Daten automatisch
# batch_size=32 ist ein guter Standardwert
dataset = TensorDataset(x_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## 4. Die Architektur (Model Class)

Jedes neuronale Netz in PyTorch ist eine Klasse, die von `nn.Module` erbt.
*   **`__init__`**: Definition der Layer (Was haben wir?).
*   **`forward`**: Definition des Datenflusses (Wie sind sie verbunden?).

```python
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        
        # --- Feature Extraction (Convolutional Part) ---
        # Layer 1: Conv -> ReLU -> Pool
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Layer 2: Conv -> ReLU -> Pool (Optional, für tiefere Netze)
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # --- Classification (Fully Connected Part) ---
        # Berechnung der Input-Größe für Linear Layer:
        # Wenn Input 64x64 war -> Pool(/2) -> 32x32 -> Pool(/2) -> 16x16.
        # Channels sind jetzt 32. Flatten Size = 32 * 16 * 16.
        self.fc = nn.Linear(32 * 16 * 16, 10) # 10 Ausgangsklassen

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        
        # Flatten: Umwandlung von 3D-Tensor (C,H,W) in 1D-Vektor für Linear Layer
        # out.size(0) ist die Batch-Size, -1 lässt PyTorch den Rest berechnen
        out = out.view(out.size(0), -1)
        
        out = self.fc(out)
        return out # Rückgabe der "Logits" (Rohwerte vor Softmax)
```

## 5. Initialisierung

```python
model = MyCNN().to(device) # Modell auf GPU schieben

# Loss Function: CrossEntropyLoss beinhaltet bereits Softmax!
criterion = nn.CrossEntropyLoss()

# Optimizer: Adam ist meist der beste Allrounder. Learning Rate (lr) oft 0.001 oder 0.01
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

## 6. Training Loop (Die Standard-Schleife)

```python
num_epochs = 10

for epoch in range(num_epochs):
    model.train() # Wichtig: Setzt Modus auf Training (für Dropout/Batchnorm relevant)
    running_loss = 0.0
    
    for images, labels in train_loader:
        # 1. Daten auf das Device schieben
        images, labels = images.to(device), labels.to(device)
        
        # 2. Gradienten zurücksetzen (WICHTIG! Sonst summieren sie sich auf)
        optimizer.zero_grad()
        
        # 3. Forward Pass (Vorhersage berechnen)
        outputs = model(images)
        
        # 4. Loss berechnen (Fehler bestimmen)
        loss = criterion(outputs, labels)
        
        # 5. Backward Pass (Gradienten berechnen - Backpropagation)
        loss.backward()
        
        # 6. Optimizer Step (Gewichte aktualisieren)
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
```

## 7. Evaluation (Testen)

```python
model.eval() # Setzt Modus auf Evaluation (deaktiviert Dropout etc.)
with torch.no_grad(): # Deaktiviert Gradientenberechnung (spart Speicher & Rechenzeit)
    correct = 0
    total = 0
    for images, labels in test_loader: # Angenommen test_loader existiert
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        # Höchsten Wert als Vorhersage nehmen
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Genauigkeit: {100 * correct / total}%')
```
