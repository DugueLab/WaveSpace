from setuptools import setup, find_packages

with open('../README.md') as f:
    long_description = f.read()

setup(
    name='WaveSpace',
    version='1.1.8',
    description='A Python package for the analysis of cortical traveling waves',
    package_dir={'': '../'},
    packages=find_packages(where='../'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/kpetras/WaveSpace',
    author='Kirsten Petras',
    author_email='kirsten.petras[at]u-paris.fr',
    license='GNU General Public License',
    classifiers=[
        'Development Status :: 4 - beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "plotly",
        "pint",
        "pyvista",
        "pandas",
        "scikit-learn",
        "scikit-image",
        "tvb-gdist",
        "emd",
        "mne"
      ]
)