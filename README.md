# Eminem, Goethe and Shakespeare sing together
## - A GPT model from scratch trained on Eminem, Goethe and Shakespeare lyrics -

  ![](/image.jpeg)

This repository contains a minimalistic implementation of the GPT model based on Andrej Karpathy's tutorial.
I extended the Dataset to also include Goethe and Eminem lyrics. The  model is therefore trained on the combined dataset of Shakespeare, Goethe and Eminem lyrics. 
Furthermore, I changed the tokenization to use the Byte-Pair-Encoding (BPE) or WordPiece tokenization (instead of the simple one from the tutorial).

### Usage
To train the model and see the data preparation, just look into the notebook 'dev.ipynb'.
The gpt.py file contains the model and the hyperparameters.

Note: this project is meant for fun, and i created it to learn more about transformers and PyTorch. 
The model is not optimized and the training is not efficient.

### Requirements
- PyTorch
- tokenizers

