# AI Library

Dieses Projekt ist eine Python-Bibliothek, die verschiedene AI-Modelle bereitstellt und eine einheitliche Trainingsschnittstelle über eine Basisklasse bietet. Die Bibliothek enthält auch Funktionen zur Datenverarbeitung.

## Struktur

Die Bibliothek ist wie folgt strukturiert:

- **ai_library/**: Hauptpaket der Bibliothek.
  - **models/**: Enthält verschiedene AI-Modelle.
  - **data_processing/**: Enthält Funktionen zur Datenverarbeitung.
  - **base_model.py**: Definiert die Basisklasse `BaseModel`.

## Installation

Um die Bibliothek zu installieren, klonen Sie das Repository und installieren Sie die Abhängigkeiten:

```bash
git clone <repository-url>
cd ai-library
pip install -r requirements.txt
```

## Verwendung

Um die Modelle und Datenverarbeitungsfunktionen zu verwenden, importieren Sie die entsprechenden Module:

```python
from ai_library.models.ai_cnn import ModelA
from ai_library.data_processing.processing1 import process_data
```

## Tests

Die Bibliothek enthält Tests, die sicherstellen, dass die Modelle und Datenverarbeitungsfunktionen korrekt funktionieren. Um die Tests auszuführen, verwenden Sie:

```bash
pytest
```

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert.