# AI Library

Dieses Projekt ist eine Python-Bibliothek, die verschiedene AI-Modelle bereitstellt und eine einheitliche Trainingsschnittstelle über eine Basisklasse bietet. Die Bibliothek enthält auch Funktionen zur Datenverarbeitung.

Ziel ist ein komplette Übersicht und eine Funktionssammlung aller wichtigen Methoden um schnell darauf zugreifen zu können.
Mitarbeit ist erwünscht

## Struktur

Die Bibliothek ist wie folgt strukturiert:

- **ai_library/**: Hauptpaket der Bibliothek.
  - **models/**: Enthält verschiedene AI-Modelle.
    - **ai_cnn.py**: Definiert das `ModelCNN`-Modell.
    - **rcnn.py**: Definiert das `RCNN`-Modell.
  - **data_processing/**: Enthält Funktionen zur Datenverarbeitung.
    - **preprocessing.py**: Enthält Funktionen zur Datenvorverarbeitung wie `normalize_data` und `remove_outliers`.
    - **augmentation.py**: Enthält Funktionen zur Datenaugmentation wie `add_noise`, `scale_data`, `flip_image` und `adjust_brightness`.
  - **base_model.py**: Definiert die Basisklasse `BaseModel`.

## Installation

Um die Bibliothek zu installieren, klonen Sie das Repository und installieren Sie die Abhängigkeiten:

```bash
git clone https://github.com/tomphible/library.git
cd ai-library
pip install -r requirements.txt
```

## Verwendung

Um die Modelle und Datenverarbeitungsfunktionen zu verwenden, importieren Sie die entsprechenden Module:

```python
from ai_library.models.ai_cnn import ModelCNN
from ai_library.models.rcnn import RCNN
from ai_library.data_processing.preprocessing import normalize_data
from ai_library.data_processing.augmentation import add_noise
```

## Tests

Die Bibliothek enthält Tests, die sicherstellen, dass die Modelle und Datenverarbeitungsfunktionen korrekt funktionieren. Um die Tests auszuführen, verwenden Sie:

```bash
pytest
```

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert.