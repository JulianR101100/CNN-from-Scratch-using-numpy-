# Fehleranalyse: Warum das ursprüngliche CNN-Modell nicht konvergierte

Dieser Bericht fasst die kritischen Fehler im ursprünglichen Code zusammen. Das Modell war zwar **syntaktisch korrekt** (es gab keine Abstürze oder Fehlermeldungen), aber **semantisch fehlerhaft**, weshalb es nicht lernen konnte und der Loss stagnierte.

## 1. Der Hauptfehler: Doppelte Normalisierung (Signal-Zerstörung)

Der gravierendste Fehler befand sich in der `forward`-Methode der `Tests.py`.

*   **Der Code:**
    ```python
    # FALSCH:
    out = conv.forward((image / 255) - 0.5)
    ```
*   **Die Ursache:** Die Funktion `generate_data()` erzeugte bereits Bilder mit Werten zwischen `0.0` und `1.0` (Floats). Die erneute Division durch `255` stauchte diese Werte auf einen winzigen Bereich zusammen (ca. `0.0` bis `0.0039`). Nach der Subtraktion von `0.5` waren **alle Inputs für das Netzwerk fast identisch** (ca. `-0.5`).
*   **Die Konsequenz:** Das Netzwerk sah praktisch "leere" Bilder ohne unterscheidbare Merkmale (Vertikal vs. Horizontal). Es gab kein Signal, das gelernt werden konnte.
*   **Warum es trotzdem lief:** Mathematisch ist die Division von kleinen Floats erlaubt. Numpy verarbeitete die Zahlen ohne Fehler weiter, auch wenn sie inhaltlich nutzlos waren.

## 2. Mathematischer Fehler: Fehlender Gradient für ReLU

*   **Der Code:**
    Im Forward-Pass wurde `np.maximum(0, out)` verwendet. Im Backward-Pass (`train`-Funktion) wurde dieser Schritt jedoch **komplett ignoriert**. Der Gradient wurde vom Pooling-Layer direkt an den Convolution-Layer weitergereicht.
*   **Die Konsequenz:** Die Nicht-Linearität wurde beim Lernen nicht berücksichtigt. Das Netzwerk versuchte, eine Funktion zu optimieren, die es gar nicht ausführte. Neuronen, die im Forward-Pass auf 0 gesetzt wurden (inaktiv), bekamen trotzdem Gradienten ab, was die Gewichte in falsche Richtungen drückte.

## 3. Suboptimale Initialisierung (Vanishing Gradients)

*   **Der Code:**
    *   Softmax: `weights = ... / input_len`
    *   Conv: `filters = ... / 9`
*   **Die Ursache:** Die Gewichte wurden zu stark verkleinert. Bei der Softmax-Schicht (Input-Länge 72) war die Standardabweichung extrem klein.
*   **Die Konsequenz:** Das Signal wurde durch die Schichten hinweg immer schwächer. Besonders beim Backpropagation wurden die Gradienten so klein ("Vanishing Gradient Problem"), dass die Gewichte sich kaum noch veränderten.

## Zusammenfassung

Das Modell war ein klassisches Beispiel für **"Silent Failures"** in neuronalen Netzen:
1.  Die **Shapes (Dimensionen)** der Tensoren stimmten überein -> Keine Matrix-Multiplikations-Fehler.
2.  Die **Syntax** war korrekt -> Keine Python Exceptions.

Das Netzwerk "riet" in jeder Epoche nur (Accuracy ~50%, Loss ~0.69), da es durch Fehler 1 keine Daten sah und durch Fehler 2 & 3 nicht effektiv lernen konnte.
