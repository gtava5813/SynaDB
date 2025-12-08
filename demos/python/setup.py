"""Setup script for Syna Python wrapper."""

from setuptools import setup, find_packages

setup(
    name="synadb",
    version="0.2.0",
    description="Python wrapper for Syna embedded database",
    author="Syna Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
    ],
    extras_require={
        "ml": [
            "torch>=2.0.0",
            "datasets>=2.14.0",
            "transformers>=4.30.0",
        ],
        "pandas": [
            "pandas>=1.3.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "hypothesis>=6.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

