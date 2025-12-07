import numpy as np
import Architecture

# Einfügen der Layer (Seperates Skript für bessere übersicht)
Conv3x3 = Architecture.Concolution3x3
MaxPool2 = Architecture.MaxPooling2x2
Softmax = Architecture.Softmax
ReLU = Architecture.ReLU


def generate_data(num_samples=100):
    """
    Erzeugt einfache 8x8 Bilder.
    Klasse 0: Vertikale Linie
    Klasse 1: Horizontale Linie
    """
    images = []
    labels = []
    for _ in range(num_samples):
        img = np.zeros((8, 8))
        label = np.random.randint(0, 2)

        if label == 0:  # Vertikal
            col = np.random.randint(0, 8)
            img[:, col] = 1  # Zeichne vertikale Linie
        else:  # Horizontal
            row = np.random.randint(0, 8)
            img[row, :] = 1  # Zeichne horizontal Linie

        # Füge etwas Rauschen hinzu (optional, macht es realistischer)
        img += np.random.randn(8, 8) * 0.1

        images.append(img)
        labels.append(label)
    return images, labels

# Globale Modell-Instanzen (werden beim Import initialisiert)
# Output Shape ist also 3x3 * 8 Filter = 72 Inputs für Softmax
conv = Conv3x3(8)
relu = ReLU()
pool = MaxPool2()
softmax = Softmax(3 * 3 * 8, 2)


def train(im, label, lr=0.01):
    # --- Forward ---
    out, loss, accuracy = forward(im, label)

    # --- Gradient Calculation (Fused Softmax+CrossEntropy) ---
    # Ki Kommentar: Dieser "y_hat - y" Gradient ist die analytisch vereinfachte 
    # Ableitung des Cross-Entropy-Loss in Kombination mit der Softmax-Funktion
    gradient = np.zeros(2)
    gradient[0] = out[0] - (1 if label == 0 else 0)
    gradient[1] = out[1] - (1 if label == 1 else 0)

    # --- Backward ---
    # übergebe den kombinierten Gradienten direkt an Softmax backward
    gradient = softmax.backward(gradient, lr)
    gradient = pool.backward(gradient)
    gradient = relu.backward(gradient) # ReLU: Propagiert Gradienten nur durch aktivierte Neuronen
    gradient = conv.backward(gradient, lr)

    return loss, accuracy


def forward(image, label):
    # Normalisierung ist wichtig! [-0.5, 0.5]
    # Ki Hinweis: generate_data erzeugt bereits Werte um 0-1 (floats), daher kein /255 nötig.
    out = conv.forward(image - 0.5)
    out = relu.forward(out)  # ReLU: Fügt Nicht-Linearität ein
    out = pool.forward(out)
    out = softmax.forward(out)

    loss = -np.log(out[label] + 1e-9)  # +epsilon für Stabilität
    accuracy = 1 if np.argmax(out) == label else 0
    return out, loss, accuracy


def run_experiment():
    """
    Führt das Training und die Evaluation durch.
    Gibt die Ergebnisse zurück, die für die Analyse/Visualisierung benötigt werden.
    """
    # --- Daten Generieren ---
    print("Generiere Daten...")
    X_train, y_train = generate_data(1000)
    X_test, y_test = generate_data(200)

    # --- Training Loop mit Tracking ---
    loss_history = []
    acc_history = []
    avg_loss_steps = []

    print("Starte Training...")
    for epoch in range(5):  # 5 Epochen sollten reichen
        print(f'--- Epoch {epoch + 1} ---')

        # Shuffle
        perm = np.random.permutation(len(X_train))
        X_train = np.array(X_train)[perm]
        y_train = np.array(y_train)[perm]

        curr_loss = 0
        curr_acc = 0

        for i, (im, label) in enumerate(zip(X_train, y_train)):
            l, acc = train(im, label, lr=0.01)
            curr_loss += l
            curr_acc += acc

            # Tracking für Plotting (alle 50 Schritte)
            if i % 50 == 0 and i > 0:
                avg_loss = curr_loss / 50
                avg_acc = curr_acc / 50
                loss_history.append(avg_loss)
                acc_history.append(avg_acc)
                avg_loss_steps.append(avg_loss)  # Nur für Print
                curr_loss = 0
                curr_acc = 0

        print(f"Avg Loss letzte 50 steps: {avg_loss_steps[-1]:.3f}")

    # --- Evaluation ---
    print("\nEvaluiere Test Set...")
    y_pred = []
    for im, label in zip(X_test, y_test):
        out, _, _ = forward(im, label)
        y_pred.append(np.argmax(out))

    # Rückgabe der Ergebnisse für Analyse.py
    # conv Objekt wird zurückgegeben, um die Filter zu visualisieren in Analyse.py
    return loss_history, acc_history, X_test, y_test, y_pred, conv


if __name__ == "__main__":
    # Wenn direkt ausgeführt: Nur Training und Text-Output, keine Plots.
    run_experiment()
