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
    affiliation: 2
  - name: Laura Dugué
    orcid: 0000-0003-3085-1458
    equal-contrib: false
    affiliation: "1, 3"

affiliations:
  - name: Université Paris Cité, INCC UMR 8002, CNRS, F-75006 Paris, France
    index: 1
  - name: Independent researcher, The Netherlands
    index: 2
  - name: Institut Universitaire de France (IUF), Paris, France
    index: 3


date: 17 February 2026
bibliography: references.bib

---
# Summary
Oscillatory cortical activity has been found to systematically propagate across space as traveling waves [@muller_cortical_2018]. Growing evidence links those spatiotemporal patterns to cognitive functions including, but not limited to visual spatial attention [@fakche2024perceptual], memory [@mohan2024direction] and consciousness [@bhattacharya2022propofol]. Detecting and characterizing traveling waves in non-invasive multichannel recordings of brain activity requires multiple processing and analysis steps, some of which are interchangeable while others can lead to diverging interpretations of the same data.
WaveSpace is a modular Python toolbox for the simulation and analysis of traveling wave dynamics in multichannel recording arrays. Based on a single, flexible data class users can execute, compare and recombine multiple commonly used analysis steps as well as evaluate their performance against simulated benchmarks. 

# Statement of need
Various approaches to detect and characterize cortical traveling waves have emerged in the literature. Typically, laboratories develop customized pipelines tailored to their experimental requirements and software platform preferences [@alexander_measurement_2006;@muller_stimulus-evoked_2014;@alamia_alpha_2019;@das_how_2022; but see also @gutzen_modular_2024 for a notable exception].

The diversity of methods and implementations found in the literature poses challenges for researchers, both in selecting the one most suitable for their own studies and in directly comparing the performance of different pipelines. WaveSpace addresses this gap by integrating commonly used strategies into a single modular framework. This framework ensures that modules for preprocessing, data decomposition, spatial arrangement of sensor positions, wave analysis, and evaluation are interchangeable within the same workflow. Additionally, a simulation module allows for the generation of benchmarking data with desired properties to directly test the accuracy and specificity of planned analysis pipelines in silico. The resulting pipelines are ready-to-use in empirical studies [@Petras2025locally;@fakche_alpha_2024].   

# State of the field
Several single purpose pipelines for the analysis of cortical traveling waves exist. Table 1 shows a non-exhaustive list of openly available analysis code. In most cases, the provided code accompanies, and is tailored towards, a single experimental contribution to the scientific literature on cortical traveling waves.  

Table 1. Single‑purpose tools for traveling‑wave analysis.

| Tool | Lang | Method | Repo |
|------|------|--------|------|
| wave-matlab | MATLAB | Phase–distance correlation | [GitHub](https://github.com/mullerlab/wave-matlab) |
| Travelling wave analysis (ScaleSymmetry) | Python | SVD phase waves | [GitHub](https://github.com/ScaleSymmetry/Traveling-wave-analysis) |
| travellingWaveEEG | MATLAB | 2D FFT | [GitHub](https://github.com/artipago/travellingWaveEEG) |
| Travelling wave analysis (jacobslab) | MATLAB | Circular–linear correlation | [GitHub](https://github.com/jacobslab/Traveling-wave-analysis) |
| NeuroPattToolbox | MATLAB | Optical flow | [GitHub](https://github.com/BrainDynamicsUSYD/NeuroPattToolbox) |
| cobrawap | Python | Optical flow (Snakemake) | [GitHub](https://github.com/NeuralEnsemble/cobrawap) |
| Phase vs Granger | MATLAB | Phase gradient + Granger | [GitHub](https://github.com/artipago/comparing-phase-based-and-Granger-based-analyses) |

In principle, multi-purpose neurophysiology data analysis packages such as fieldtrip [@oostenveld2011fieldtrip] for Matlab or MNE [@gramfort2014mne] for Python could be extended to include traveling wave analysis methods. Given the wide variety of available approaches and the lack of systematic comparison in the literature, WaveSpace was instead designed as a standalone tool with consistent workflows dedicated exclusively to traveling wave analysis. However, to ensure users can still benefit from the pre-processing, time-frequency decomposition and visualization methods provided by MNE, WaveSpace easily integrates MNE data objects at any stage of processing. 

# Functionality and software design
WaveSpace has been developed to provide an array of methods for the detection and analysis of cortical traveling waves, primarily in non-invasive electrophysiology data such as electro- or magnetoencephalography. All WaveSpace functionality is based on a single data class, called WaveData, that enforces conventions for data dimension order and dimension naming. This allows for most processing steps to be interchangeable. The WaveData class organizes the in- and output of consecutive processing steps into discrete data-buckets, while logging progress. The entire framework is comprehensively documented and includes example scripts to facilitate adoption.

WaveSpace contains 6 core modules (see figure 1 for module overview):

- Decomposition: Provides multiple techniques to decompose broadband data into frequency components, including FFT-based methods (e.g., wavelets, filter-Hilbert), empirical mode decomposition (EMD), and generalized phase analysis.

- Spatial Arrangement: Includes methods to map 3D sensor positions onto 2D regular grids using approaches such as multidimensional scaling (MDS) and isomap. Multiple interpolation options are available.

- Wave Analysis: Offers a variety of analysis methods, such as 2D FFT, optical flow analysis, phase gradient methods, and principal component analysis (PCA).

- Simulation: Functions to simulate traveling and spatially stationary (i.e., standing) waves with both linear and nonlinear properties, as well as incorporate noise.

- Evaluation and statistics: Summarizes wave characteristics. Contains options for wave-scoring. 
 
- Plotting: Contains visualization tools for each analysis option.

![WaveSpace Module Overview](WaveSpace_overview.png)
*Figure 1: Overview of WaveSpace modules.*

# Research impact statement
WaveSpace has been publically introduced during a workshop at the 47th European conference on visual perception (2025) and used in peer reviewed as well as ongoing work [@Petras2025locally;@fakche_alpha_2024;@kong2025oscillatory]. Users are invited to contribute to the ongoing package development via github.  

# Funding
This project received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation program (grant agreement No. 852139 - Laura Dugué).

# Toolbox dependencies
[Environment file](https://github.com/kpetras/WaveSpace/blob/main/WaveSpaceEnv.yaml)

# AI usage disclosure
Github copilot in "ask" mode has been used in the initial translation of Matlab code to Python. When the resulting code was found to be inaccurate and failed to match the style of the rest of the package, most of it was manually re-written. Single word autocomplete was used throughout for code and comments. Copilot was used for code and formatting suggestions. No agentic AI was used. 

# References
