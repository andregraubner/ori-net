
# Ori-Net

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

Ori-Net is a minimalistic PyTorch implementation for fine-tuning [Caduceus](https://github.com/kuleshov-group/caduceus) to predict plasmid Origins of Replication (ORIs).

## Overview

This proof-of-concept demonstrates that pre-trained DNA language models can predict origins of replication by casting the problem as a span-prediction task, similar to BERT question answering. The implementation is intentionally simple and does not provide extensive abstractions or utilities for using the model.

## Getting Started

### Data Preparation

1. Download plasmid ORI data from [DoriC](https://tubic.org/doric/) and place it at `dataset/annotations.csv`.
2. Download the corresponding sequences (e.g., from [NCBI](https://www.ncbi.nlm.gov/guide/howto/dwn-records/)) and place them at `dataset/sequence.fasta`.

### Running the Model

To train the model, simply run:```python train_caduceus.py```

## Contributing

Feel free to contribute by adding abstractions, utilities, or improvements to the existing codebase. Pull requests are welcome!

## Acknowledgements

Special thanks to the following open-source projects:

- [Caduceus](https://github.com/kuleshov-group/caduceus)
- [lucidrains](https://github.com/lucidrains)

## Citation

If you use this code in your research, please cite:```bibtex
@misc{ori-net,
  author = {Andre Graubner},
  title = {Ori-Net: Fine-tuning Caduceus for Plasmid ORI Prediction},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/andregraubner/ori-net}}
}
```

---

**Note:** This is a minimalistic implementation intended for research purposes. For production use, additional testing and optimization may be required.
