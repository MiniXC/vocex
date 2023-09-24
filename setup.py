from setuptools import setup, find_packages

setup(
    name="vocex",
    version="0.1.10",
    description="Voice Frame-Level and Utterance-Level Attribute Extraction",
    author="Christoph Minixhofer",
    author_email="christoph.minixhofer@gmail.com",
    packages=find_packages(),
    install_requires=[
        "torchaudio>=0.8.0",
        "librosa>=0.8.0",
        "numpy>=1.19.0",
        "tqdm>=4.48.2",
        "transformers>=4.0.0",
    ],
    python_requires=">=3.6",
    license="MIT",
)

# training on tpus seems to be broken on accelerate 0.20+, 0.19.0 works
