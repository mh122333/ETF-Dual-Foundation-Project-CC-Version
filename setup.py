"""Setup script for etf_pipeline package."""

from setuptools import setup, find_packages

setup(
    name="etf_pipeline",
    version="0.1.0",
    description="ETF Duel Foundation Model - Experiment 0 Pipeline",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "alpaca-py>=0.10.0",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "pyarrow>=10.0.0",
        "scikit-learn>=1.0.0",
        "pytz>=2022.1",
    ],
)
