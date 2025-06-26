from setuptools import setup, find_packages

setup(
    name="kwantlib",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "plotly",
        "ray",
        ],
    author="Pierre Renucci",
    author_email="renuccip@icloud.com",
    description="Une bibliothèque pour l'analyse quantitative",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/cnernc/kwantlib",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
