from setuptools import setup

setup(
    name='fkpwin',
    version='0.2.0',
    description='Window functions for FKP-like estimators of Fourier-space correlation functions',
    author="Pierre Zhang",
    license='MIT',
    packages=['fkpwin'],
    install_requires=['numpy', 'scipy', 'pyyaml', 'astropy', 'mpmath', 'fftlog @ git+https://github.com/pierrexyz/fftlog'],
    package_dir = {'fkpwin': 'fkpwin'},
    zip_safe=False,

    classifiers = [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Programming Language :: Python",
    ],
)
