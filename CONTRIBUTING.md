# Contributing to Cyclical SG-MCMC

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the Cyclical Stochastic Gradient MCMC implementation.

## 🚀 Getting Started

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/BahaMelki0/cyclical-sgmcmc.git
   cd cyclical-sgmcmc
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

5. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## 🔧 Development Workflow

### Creating a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### Making Changes

1. **Write your code** following the project's coding standards
2. **Add tests** for new functionality
3. **Update documentation** if needed
4. **ADD tests** in /tests
5. **Run tests** to ensure everything works:
   ```bash
   pytest tests/
   ```

### Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **pytest** for testing

Run all checks:
```bash
# Format code
black cyclical_sgmcmc/ experiments/ 
isort cyclical_sgmcmc/ experiments/ 

# Check linting
flake8 cyclical_sgmcmc/ experiments/ 

# Run tests
pytest tests/ --cov=cyclical_sgmcmc
```

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add cyclical learning rate scheduler
fix: resolve memory leak in SGHMC sampler
docs: update installation instructions
test: add unit tests for uncertainty metrics
```

## 📝 Types of Contributions

### 🐛 Bug Reports

When reporting bugs, please include:

- **Clear description** of the issue
- **Steps to reproduce** the problem
- **Expected vs actual behavior**
- **Environment details** (Python version, PyTorch version, OS)
- **Error messages** or stack traces

### ✨ Feature Requests

For new features, please provide:

- **Clear description** of the proposed feature
- **Use case** and motivation
- **Possible implementation** approach
- **Backward compatibility** considerations

### 🔬 Research Contributions

We welcome research-oriented contributions:

- **New MCMC variants** or improvements
- **Additional experiments** or benchmarks
- **Theoretical analysis** or proofs
- **Performance optimizations**

### 📚 Documentation

Documentation improvements are always welcome:

- **API documentation** improvements
- **Tutorial notebooks** or examples
- **Theory explanations** or background
- **Installation guides** for different platforms

## 🧪 Testing Guidelines

### Writing Tests

- **Unit tests** for individual functions/classes
- **Integration tests** for complete workflows
- **Regression tests** for bug fixes
- **Performance tests** for optimization claims

### Test Structure

```python
import pytest
import torch
from cyclical_sgmcmc import CyclicalSGLD

class TestCyclicalSGLD:
    def test_stepsize_schedule(self):
        """Test that stepsize follows cyclical pattern."""
        # Test implementation
        pass
    
    def test_stage_transitions(self):
        """Test exploration/sampling stage transitions."""
        # Test implementation
        pass
```

## 📊 Benchmarking

When adding new features or optimizations:

1. **Benchmark against baselines** using existing experiments
2. **Document performance changes** in pull request
3. **Include timing comparisons** for significant changes
4. **Test on multiple hardware** configurations if possible

## 🔍 Code Review Process

### Pull Request Guidelines

1. **Create descriptive PR title** and description
2. **Reference related issues** using keywords (fixes #123)
3. **Include test results** and benchmarks
4. **Update documentation** as needed
5. **Ensure CI passes** all checks

### Review Checklist

- [ ] Code follows project style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance impact is acceptable
- [ ] Code is well-commented and readable

## 🏗️ Project Structure

Understanding the codebase:

```
├── samplers/           # Core MCMC implementations
├── models/             # Neural network architectures
└── utils/              # Utility functions

experiments/            # Reproduction experiments
├── synthetic/          # Synthetic data experiments
├── bnn_classification/ # Classification experiments
└── uncertainty/        # Uncertainty estimation

```

## 🎯 Contribution Ideas

Looking for ways to contribute? Here are some ideas:

### Beginner-Friendly
- Fix typos in documentation
- Add type hints to functions
- Improve error messages
- Add more unit tests

### Intermediate
- Implement additional MCMC variants
- Add support for new datasets
- Optimize memory usage
- Create tutorial notebooks

### Advanced
- Theoretical analysis of convergence
- Distributed/parallel implementations
- Integration with other frameworks
- Novel uncertainty quantification methods

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Email**: [bahaeddine.melki@eurecom.fr] for private inquiries

## 🙏 Recognition

Contributors will be:

- **Listed in CONTRIBUTORS.md**
- **Mentioned in release notes**
- **Credited in academic papers** (for significant contributions)

## 📜 Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). Please be respectful and inclusive in all interactions.

---

Thank you for contributing to advancing Bayesian deep learning research! 🚀

