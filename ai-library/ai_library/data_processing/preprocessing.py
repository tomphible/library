import numpy as np
from sklearn.preprocessing import StandardScaler

def normalize_data(data):
    """Normiert die Daten auf Mittelwert 0 und Varianz 1"""
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def remove_outliers(data, threshold=3.0):
    """Entfernt Ausreißer basierend auf Standardabweichung"""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    filtered_data = data[np.abs(data - mean) < threshold * std]
    return filtered_data