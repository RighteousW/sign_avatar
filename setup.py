"""
Setup configuration for NSL Translator project.
"""

from pathlib import Path
from setuptools import setup, find_packages


def parse_requirements(filename):
    """
    Parse a requirements.txt file and return a list of dependencies.

    Filters out:
    - Comments (lines starting with #)
    - Empty lines
    - Editable installs (-e)
    - Direct file references

    Args:
        filename: Path to requirements.txt file

    Returns:
        List of requirement strings
    """
    requirements = []
    filepath = Path(__file__).parent / filename

    if not filepath.exists():
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Skip editable installs
            if line.startswith("-e"):
                continue

            # Skip direct file references
            if line.startswith("file://") or line.startswith("."):
                continue

            # Skip other pip flags
            if line.startswith("-"):
                continue

            requirements.append(line)

    return requirements


setup(
    name="nsl-translator",
    version="0.1.0",
    description="NSL Bidirectional Translator using Avatar",
    author="Righteous Wasambo",
    author_email="wasambor@gmail.com",
    url="https://github.com/RighteousWasambo/sign_avatar",
    packages=find_packages(where="."),
    python_requires=">=3.12.3",
    install_requires=parse_requirements("requirements.txt"),
    entry_points={
        "console_scripts": [
            # GUI applications
            "nsl-gui=src.gui.integrated_gui:main",
        ],
    },
)
