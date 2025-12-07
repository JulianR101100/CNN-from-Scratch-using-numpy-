import numpy as np


class Concolution3x3:
    """
    Ein Convolution Layer mit 3x3 Filtern
    8 Filter für die Merkmalsextraktion
    """

    def __init__(self, num_filters):
        self.num_filters = num_filters
        # Initialisierung der Filter
        # Wir nutzen np.random.randn(...) / 3, was etwa Xavier entspricht (std = 1/sqrt(9) = 1/3)
        self.filters = np.random.randn(num_filters, 3, 3) / 3

    def iterate_regions(self, image):
        """
        Generator: Schiebt das 3x3 Fenster über das Bild (Valid Padding).
        Das Bild wird um 2 Pixel kleiner (z.B. 28x28 -> 26x26).
        """
        h, w = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                image_region = image[i:(i + 3), j:(j + 3)]
                yield image_region, i, j

    def forward(self, input):
        """
        Die Vorwärts-Faltung: Z = X * K
        """
        self.last_input = input  # Wir brauchen den Input X für Backprop!
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for image_region, i, j in self.iterate_regions(input):
            # Das Herzstück: Skalarprodukt zwischen Bildausschnitt und ALLEN Filtern
            # Wir machen das parallel für alle Filter dank numpy broadcasting
            output[i, j] = np.sum(image_region * self.filters, axis=(1, 2))

        return output

    def backward(self, d_L_d_out, learn_rate):
        """
        Backpropagation für den Conv Layer.
        d_L_d_out ist das 'delta', das vom nächsten Layer (Pooling/ReLU) zurückkommt.
        """
        d_L_d_filters = np.zeros(self.filters.shape)

        # Iterriere erneut über das Bild, um zu sehen, welcher Input 
        # zu welchem Fehler beigetragen hat.
        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                # KI korrektur IHRE FORMEL: d_L_d_K = sum(delta * X)
                # d_L_d_out[i, j, f] ist das skalare delta an Position i,j für Filter f
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        # Update der Gewichte (Gradient Descent)
        # K_neu = K_alt - learning_rate * Gradient
        self.filters -= learn_rate * d_L_d_filters

        # KI Hinweis: Normalerweise müssten wir hier auch d_L_d_input berechnen
        # und zurückgeben, falls noch ein Layer davor wäre. Da dies der erste Layer ist,
        # In diesem Code ist der Convolution-Layer der erste Layer im Netzwerk.
        # Es gibt keinen Layer davor, der ein Fehlersignal bräuchte
        # Deshalb reicht es, nur die Filter zu aktualisieren.
        return None


class MaxPooling2x2:
    """
    Max Pooling Layer mit Fenstergröße 2x2.
    """

    def iterate_regions(self, image):
        """
        Generator: Schiebt 2x2 Fenster (non-overlapping).
        """
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input
        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            # Maximum über die Achsen 0 und 1 (räumlich), Filter-Achse bleibt
            output[i, j] = np.amax(im_region, axis=(0, 1))

        return output

    def backward(self, d_L_d_out):
        """
        Leitet den Gradienten nur an das Pixel weiter, das im Forward-Pass das Maximum war.
        """
        d_L_d_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # Wenn dieses Pixel das Maximum war, bekommt es den Gradienten
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

        return d_L_d_input


class ReLU:
    def forward(self, input):
        self.last_input = input
        return np.maximum(0, input)

    def backward(self, d_L_d_out):
        d_L_d_input = d_L_d_out.copy()
        d_L_d_input[self.last_input <= 0] = 0
        return d_L_d_input


class Softmax:
    def __init__(self, input_len, nodes):
        # He-Initialization / Xavier für bessere Startwerte
        # Wir teilen durch np.sqrt(input_len), um die Varianz zu erhalten (Standard Xavier)
        self.weights = np.random.randn(input_len, nodes) / np.sqrt(input_len)
        self.biases = np.zeros(nodes)

    def forward(self, input):
        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input

        input_len, nodes = self.weights.shape
        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals

        # Numerisch stabiler Softmax (Subtraktion des Max-Werts verhindert Overflow)
        exp_vals = np.exp(totals - np.max(totals))
        return exp_vals / np.sum(exp_vals, axis=0)

    def backward(self, d_L_d_out, learn_rate):
        """
        Berechnet Gradienten für diesen Layer.
        d_L_d_out: Der Gradient vom Loss (Form: 2,) -> entspricht (y_hat - y)
        """
        # Wir wissen:
        # Input (x) Form: (72,)  (geflattet)
        # Weights (W) Form: (72, 2)
        # d_L_d_out (delta) Form: (2,)

        # 1. Gradient bezüglich der Gewichte (dW)
        # Formel: dL/dW = input^T * delta
        # Wir machen aus den Vektoren 2D-Matrizen für die Multiplikation:
        # (72, 1) @ (1, 2) -> (72, 2)
        d_L_d_w = self.last_input[np.newaxis].T @ d_L_d_out[np.newaxis]

        # 2. Gradient bezüglich der Biases (db)
        # Formel: dL/db = delta
        d_L_d_b = d_L_d_out

        # 3. Gradient bezüglich des Inputs (dX) -> Das geben wir zurück an den Pool-Layer
        # Formel: dL/dX = W * delta
        # (72, 2) @ (2,) -> (72,)
        d_L_d_inputs = self.weights @ d_L_d_out

        # 4. Update der Parameter (Gradient Descent)
        self.weights -= learn_rate * d_L_d_w
        self.biases -= learn_rate * d_L_d_b

        # 5. Zurückformen in 3D für den vorherigen Layer (z.B. 3x3x8)
        return d_L_d_inputs.reshape(self.last_input_shape)