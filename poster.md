# Evidence Detection Using Deep Learning Techniques

## Task description

## Dataset description

## Method B -- LSTM 

### Overview
1. Embedding: Tokens are converted to vectors using pretrained FastText 300d embeddings (loaded via gensim). Simple preprocessing (lowercase, regex cleanup).
2. Encoding: Both the claim and evidence are independently encoded by a shared BiLSTM encoder (configurable layers/hidden size, uses packed sequences for variable-length inputs).
3. Co-Attention: Computes cross-attention between claim and evidence hidden states, producing enhanced representations that capture interactions: [h, ctx, h-ctx, h*ctx] (concatenation of original, attended,
  difference, and element-wise product).
4. Composition: The enhanced representations are projected down and passed through a second BiLSTM for composition.
5. Pooling: Mean + max pooling over the composed sequences for both claim and evidence, concatenated into a fixed-size vector (B, 4D).
6. Classification Head: Swappable heads — mlp (2-layer with LayerNorm/GELU), linear, or deep_mlp (3-layer with residual connection). Outputs a single logit for binary classification.

### Results

## Method C -- Transformer 

### Overview

### Results

## Error Analysis

## Limitations and Ethical Considerations

## Figures and Tables
 - dev results table for mdoel B
 - dev results table for model C
 - Model B architecture 
 - Model C architecture 



