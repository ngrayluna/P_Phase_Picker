<!-- ---
# P_Phase_Picker
--- -->

<img src="./imgs/Neurons-Network_T.jpg">

<p align ="center"> 
    <font face='Helvetica'><h1><b>Earthquake Phase Picking</b><</h1></font><br>
    <font face='Helvetica'>Email: nluna{at}berkeley.edu</font>
</p>

<img src="./imgs/Neurons-Network_B.jpg">  
---

<h2>Table of Content</h2>
[What is This?](#what)  
[About](#about)  
[Dependencies](#dependencies)  
[How to Use](#how)  
[Further Reading](#reading)  
[Links](#links)  


<h2><a name="what">What is This?</a></h2>  
Supplementary source code for “Identifying Seismic Waves with Convolutional Neural Networks” tutorial on my website:  

[insert website here].  
[insert screen shot of blog here]  

<h2><a name="about">About</a></h2>  
The following is one approach of how to train a Convolutional Neural Network to identify the first phase (P-Phase) of an earthquake using time-series data measured on a seismometer. This repository serves as a starting point for those who benefit from seeing an example, from start to finish, of how to using seismic data for training a 1D convolutional neural network. This model has room for improvement and should serve only as a starting point. 

The machine learning workflow stages included are:  

Downloading Seismic Data, Pre-Processing the Data (& Formatting), Defining and Compiling a 1D Convolutional Neural Network, and Training.  

<h2><a name="dependencies">Dependencies:</a></h2>  
All scripts are written in Python3. The following libraries (and their version) were also used:  
* NumPy 1.15.3  
* ObsPy 1.1.0  
* Pandas 0.23.0  
* Keras (on top of TensorFlow)  2.2.4  
* TensorFlow 1.8.0  
* Matplotlib  

<h2><a name="how">How to Use</a></h2>  
Run the following .py scripts in the following order in order to replicate what was done for the tutorial:  

* /bin/download_data/define_stations.py  
* /bin/download_data/get_quakefiles.py  
* /bin/download_data/get_mass_picks.py  
* /bin/download_data/get_mass_data.py [currently not included]  
* / bin/pre_process/make_data_set.py  
* /bin/pre_process/format_waveforms.py  
* /models/p_phase_picker.py  


Make note of where each output directory is stored relative to your home directory.  

<h2><a name="links">Links</a></h2>  
* Web Site  
* <a href = "https://github.com/ngrayluna/P_Phase_Picker">Source Code</a>  



