# Transformer Architecture Implementation

A PyTorch implementation of the Transformer architecture from "Attention Is All You Need" (2017). This implementation includes all core components of the Transformer model with detailed explanations and examples.

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/transformer_project.git
cd transformer_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
```
transformer_project/
├── README.md           # This file - project setup and usage
├── CONCEPTS.md         # Detailed explanations and concepts
├── requirements.txt    # Project dependencies
├── src/
│   ├── __init__.py
│   ├── attention.py    # Self-attention and Multi-head attention
│   ├── encoder.py      # Encoder implementation
│   ├── decoder.py      # Decoder implementation
│   ├── transformer.py  # Complete transformer model
│   └── utils.py        # Utility functions
└── examples/
    └── translation_example.py
```

## Features

- Complete implementation of the Transformer architecture
- Modular design for easy understanding and modification
- Support for both training and inference
- Detailed documentation and examples
- Configurable hyperparameters

## Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch 1.9.0 or higher
- CUDA (optional, for GPU support)

### Dependencies
Install all required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Example
```python
from src.transformer import Transformer

# Initialize transformer
transformer = Transformer(
    src_vocab_size=10000,
    trg_vocab_size=10000,
    src_pad_idx=0,
    trg_pad_idx=0,
    embed_size=256,
    num_layers=6,
    forward_expansion=4,
    num_heads=8,
    dropout=0.1,
    max_length=100
)

# Forward pass
output = transformer(src, trg)
```

### Training Example
```python
import torch
from src.transformer import Transformer

# Initialize model and optimizer
model = Transformer(...)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        src, trg = batch
        output = model(src, trg)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
```

## Model Architecture

### Components
1. **Self-Attention**
   - Scaled dot-product attention
   - Multi-head attention mechanism
   - Attention masking

2. **Encoder**
   - Stack of N identical layers
   - Each layer has:
     - Multi-head self-attention
     - Position-wise feed-forward network
     - Layer normalization
     - Residual connections

3. **Decoder**
   - Stack of N identical layers
   - Each layer has:
     - Masked multi-head self-attention
     - Multi-head cross-attention
     - Position-wise feed-forward network
     - Layer normalization
     - Residual connections

### Positional Encoding
- Adds position information to input embeddings
- Uses sine and cosine functions
- Helps model understand word order
- Example:
```
Position 1: [0.0, 1.0, 0.0, 1.0, ...]
Position 2: [0.84, 0.54, 0.84, 0.54, ...]
Position 3: [0.91, 0.41, 0.91, 0.41, ...]
```

## Configuration

### Hyperparameters
- `src_vocab_size`: Source vocabulary size
- `trg_vocab_size`: Target vocabulary size
- `embed_size`: Embedding dimension
- `num_layers`: Number of encoder/decoder layers
- `num_heads`: Number of attention heads
- `forward_expansion`: Feed-forward network expansion factor
- `dropout`: Dropout rate
- `max_length`: Maximum sequence length

### Training Parameters
- Learning rate: 3e-4 (default)
- Batch size: 32 (default)
- Warmup steps: 4000 (default)
- Label smoothing: 0.1 (default)
- Gradient clipping: 1.0 (default)

## Best Practices

1. **Training**
   - Use learning rate warmup
   - Apply gradient clipping
   - Use label smoothing
   - Implement early stopping

2. **Performance**
   - Use mixed precision training
   - Implement efficient attention
   - Optimize data loading

## Common Issues and Solutions

1. **Memory Issues**
   - Reduce batch size
   - Use gradient accumulation
   - Implement model parallelism

2. **Training Stability**
   - Use proper initialization
   - Apply layer normalization
   - Use learning rate warmup

3. **Performance Optimization**
   - Use torch.jit for model optimization
   - Implement efficient attention mechanisms
   - Use data prefetching
   - Enable CUDA optimizations

## Advanced Features

1. **Model Optimization**
   - Quantization support
   - Pruning capabilities
   - Knowledge distillation
   - Model parallelism

2. **Training Enhancements**
   - Mixed precision training
   - Gradient accumulation
   - Learning rate scheduling
   - Model checkpointing

3. **Inference Optimizations**
   - Beam search decoding
   - Length penalty
   - Temperature scaling
   - Top-k and nucleus sampling

## Documentation

For detailed explanations of concepts, mechanisms, and mathematics, please refer to `CONCEPTS.md`.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original Transformer paper: "Attention Is All You Need" (2017)
- PyTorch documentation and community
- Contributors and maintainers

## Contact

For questions and feedback, please open an issue in the repository.

## Citation

If you use this implementation in your research, please cite:
```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, Lukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

## Multi-Head Attention

### The Concept
Instead of one set of eyes, imagine having multiple sets:
- Each set looks at the text differently
- One might focus on grammar
- Another on meaning
- Another on context

### Benefits
1. **Multiple Perspectives**: Different heads learn different aspects
2. **Better Understanding**: Combined knowledge from all heads
3. **More Robust**: Less likely to miss important information
