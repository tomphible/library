import numpy as np
import random
from typing import List
from PIL import Image, ImageEnhance
import nltk
from nltk.tokenize import word_tokenize

# Falls NLTK noch nicht initialisiert wurde:
nltk.download('punkt')

### 🔹 1. Augmentation für numerische Daten ###
def add_noise(data: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
    """
    Fügt Rauschen zu numerischen Daten hinzu.
    
    :param data: NumPy-Array mit numerischen Daten.
    :param noise_level: Stärke des Rauschens (Standardabweichung).
    :return: NumPy-Array mit Rauschen.
    """
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def scale_data(data: np.ndarray, scale_factor: float = 1.1) -> np.ndarray:
    """
    Skaliert numerische Daten um einen bestimmten Faktor.
    
    :param data: NumPy-Array mit numerischen Daten.
    :param scale_factor: Skalierungsfaktor (>1 verstärkt, <1 verkleinert).
    :return: Skaliertes NumPy-Array.
    """
    return data * scale_factor


### 🔹 2. Augmentation für Bilddaten ###
def flip_image(image: Image.Image, horizontal: bool = True) -> Image.Image:
    """
    Spiegelt ein Bild horizontal oder vertikal.
    
    :param image: PIL.Image-Objekt.
    :param horizontal: Falls True, horizontale Spiegelung, sonst vertikal.
    :return: Transformiertes Bild.
    """
    if horizontal:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        return image.transpose(Image.FLIP_TOP_BOTTOM)

def adjust_brightness(image: Image.Image, factor: float = 1.2) -> Image.Image:
    """
    Passt die Helligkeit eines Bildes an.
    
    :param image: PIL.Image-Objekt.
    :param factor: Helligkeitsfaktor (>1 heller, <1 dunkler).
    :return: Helligkeitsangepasstes Bild.
    """
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


### 🔹 3. Augmentation für Textdaten ###
def synonym_replacement(text: str, num_words: int = 2) -> str:
    """
    Ersetzt zufällige Wörter im Text durch Synonyme.
    
    :param text: Eingabetext als String.
    :param num_words: Anzahl der zu ersetzenden Wörter.
    :return: Modifizierter Text.
    """
    words = word_tokenize(text)
    if len(words) < num_words:
        return text  # Zu wenige Wörter zum Ersetzen

    new_words = words[:]
    for _ in range(num_words):
        idx = random.randint(0, len(words) - 1)
        synonyms = get_synonyms(words[idx])
        if synonyms:
            new_words[idx] = random.choice(synonyms)

    return " ".join(new_words)

def get_synonyms(word: str) -> List[str]:
    """
    Holt Synonyme eines Wortes (Platzhalterfunktion, kann mit WordNet erweitert werden).
    
    :param word: Eingabewort.
    :return: Liste von Synonymen.
    """
    # Hier könnte NLTK WordNet oder eine andere API genutzt werden
    synonym_dict = {
        "gut": ["toll", "super", "fantastisch"],
        "schnell": ["flink", "rasch", "fix"],
        "klug": ["intelligent", "weise", "clever"]
    }
    return synonym_dict.get(word, [])

