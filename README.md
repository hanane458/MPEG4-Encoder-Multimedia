# Mini encodeur video simplifie (TP)

## Structure

- `main.py`: pipeline complet encode/decode + metriques
- `preprocessing.py`: RGB -> YCbCr
- `intra.py`: I-frame (DCT, quantification, reconstruction)
- `inter.py`: P-frame (vecteurs de mouvement + residu)
- `entropy.py`: RLE + sauvegarde binaire `.bin`
- `decoder.py`: reconstruction de la sequence
- `utils.py`: fonctions utilitaires
- `frames/`: images d'entree

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Execution

Mettre 10 a 30 images dans `frames/`personnelement on a fait 20 , puis:

```bash
python main.py --frames_dir frames --output_bin output.bin --gop 5 --quality_scale 1.0
```

## Sorties

- `output.bin`: donnees compressees
- `reconstructed/`: images reconstruites
- `visualisation.png`: image de comparaison
