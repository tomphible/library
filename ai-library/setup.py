from setuptools import setup, find_packages

setup(
    name='ai-library',
    version='0.1.0',
    author='Tom-Philipp Bleich',
    author_email='tomphischoe@gmail.com',
    description='Eine Bibliothek zur Bereitstellung verschiedener AI-Modelle und Datenverarbeitungsfunktionen.',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "torch"
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)