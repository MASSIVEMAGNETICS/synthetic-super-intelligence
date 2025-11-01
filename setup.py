"""Setup configuration for SSI Research System."""

from setuptools import setup, find_packages

setup(
    name="ssi-codex",
    version="0.1.0",
    description="Synthetic Super Intelligence Research System",
    author="MASSIVEMAGNETICS",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "networkx>=3.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "ssi-codex=ssi_codex.cli:main",
        ],
    },
)
