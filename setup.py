#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="cyclical-sgmcmc",
    version="1.0.0",
    author="Bahaeddine Melki",
    author_email="Bahaeddine.melki@eurecom.fr",
    description="Cyclical Stochastic Gradient MCMC for Bayesian Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bahamelki0/cyclical-sgmcmc",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.1",
        "numpy>=1.19.2",
        "matplotlib>=3.3.2",
        "scipy>=1.5.2",
        "scikit-learn>=0.23.2",
        "tqdm>=4.50.2",
        "pandas>=1.1.3",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
            "pre-commit>=2.0",
            "sphinx>=3.0",
            "sphinx-rtd-theme>=0.5",
        ],
        "experiments": [
            "jupyter>=1.0",
            "notebook>=6.0",
            "ipywidgets>=7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cyclical-sgmcmc-synthetic=experiments.synthetic.run_synthetic:main",
            "cyclical-sgmcmc-cifar=experiments.bnn_classification.run_cifar:main",
            "cyclical-sgmcmc-uncertainty=experiments.uncertainty.run_uncertainty:main",
        ],
    },
    include_package_data=True,
    package_data={
        "cyclical_sgmcmc": ["*.txt", "*.md"],
    },
    keywords="bayesian deep-learning mcmc pytorch machine-learning uncertainty",
    project_urls={
        "Bug Reports": "https://github.com/BahaMelki0/cyclical-sgmcmc/issues",
        "Source": "https://github.com/BahaMelki0/cyclical-sgmcmc",
        "Documentation": "https://cyclical-sgmcmc.readthedocs.io/",
    },
)

