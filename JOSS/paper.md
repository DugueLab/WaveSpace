---
title: 'WaveSpace: analysis and simulation of cortical traveling waves'

tags:
  - Python
  - cortical traveling waves
  - simulation
  - phase-gradient
  - optical flow analysis
  - PCA 
  - circular-linear correlation
  
  
authors:
  - name: Kirsten Petras
    orcid: 0000-0001-5865-921X
    affiliation: 1
  - name: Dennis Croonenberg
    orcid: 0009-0001-4303-9306
    equal-contrib: false
    affiliation: 2
  - name: Laura Dugué
    orcid: 0000-0003-3085-1458
    equal-contrib: false
    affiliation: "1, 3"

affiliations:
  - name: Université Paris Cité, INCC UMR 8002, CNRS, F-75006 Paris, France
    index: 1
  - name: Independent Researcher, The Netherlands
    index: 2
  - name: Institut Universitaire de France (IUF), Paris, France
    index: 3


date: 15 September 2025
bibliography: references.bib

---


# Statement of need

Oscillatory cortical activity has been found to systematically propagate across space [@muller_cortical_2018]. Various approaches to detect and characterize such spatiotemporal patterns of activity, often referred to as "oscillatory cortical traveling waves (ocTW)," have emerged in the literature. Typically, laboratories develop customized pipelines tailored to their experimental requirements and software platform preferences [@alexander_measurement_2006; @muller_stimulus-evoked_2014; @alamia_alpha_2019;@das_how_2022; but see also @gutzen_modular_2024 for a notable exception].

The diversity of methods and implementations found in the literature poses challenges for researchers, both in selecting the one most suitable for their own studies and in directly comparing the performance of different pipelines. WaveSpace addresses this gap by integrating commonly used strategies into a single modular framework. This framework ensures that modules for preprocessing, data decomposition, spatial arrangement of sensor positions, wave analysis, and evaluation are interchangeable within the same workflow. The resulting pipelines are ready-to-use in empirical studies [@petras_locally_2025;@fakche_alpha_2024].   

# Functionality
WaveSpace contains 5 modules (see figure 1 for module overview):

- Decomposition: Provides multiple techniques to decompose broadband data into frequency components, including FFT-based methods (e.g., wavelets, filter-Hilbert), empirical mode decomposition (EMD), and generalized phase analysis.

- Spatial Arrangement: Includes methods to map 3D sensor positions onto 2D regular grids using approaches such as multidimensional scaling (MDS) and isomap. Multiple interpolation options are available.

- Wave Analysis: Offers a variety of analysis methods, such as 2D FFT, optical flow analysis, phase gradient methods, and principal component analysis (PCA).

- Simulation: Functions to simulate traveling and spatially stationary (i.e., standing) waves with both linear and nonlinear properties, as well as incorporating noise.

- Plotting: Contains visualization tools for each analysis option.

The entire framework is comprehensively documented and includes example scripts to facilitate its adoption.

![WaveSpace Module Overview](WaveSpace_overview.png)
*Figure 1: Overview of WaveSpace modules.*

# Funding

This project received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation program (grant agreement No. 852139 - Laura Dugué).

# Toolbox dependencies

Python (3.9.21)
mne (1.8.0)
plotly (5.24.1)  
pint (0.24.4)  
pyvista (0.44.2)  
chaospy(4.3.19)
pandas (2.2.3)  
scikit-learn (1.6.1)  
scikit-image (0.18.1)
tvb-gdist (2.2.1)  
emd (0.8.0)

# References
