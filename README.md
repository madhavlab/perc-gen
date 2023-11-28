# perc-gen

This work focuses on developing a percussion accompaniment generator that a singer can easily control to synchronously follow her tempo, with room for improvisations.
We have developed a mathematical framework for percussion generation and integrated it with a sensor system for real-time control of its tempo in an ergonomic way. This
framework uses finite-state transducers and allows for improvisations. The framework is also flexible to integrate new percussion instruments and new rhythm patterns in 
the genre of user’s choice. 
This repository contains the codes for this framework, Demo Videos and Audios that can be used for subjective evaluation and some Experimental Data and  results.

**Metronome**

This Folder contain all the codes necessary for running the system. 
**Arduino** contains the code to be uploaded and run in ana Arduino IDE. The sensor requirements will be as mentioned in the paper.
**CSV** Folder contains the information about Strokes, Beat Cycles, Call Cycles and Fillers. This information can be edited by the user. The user can Add/ Edit/ Delete customised Beat Cycles and Call Cycles. 
**WAV** Folder contains the pre recorded Stroke audios.

The other files contain the code for the system written in python. 
**metronome_kl.py** contains the system implemented by using Kalman Filters instead of our control algorithm.
**Kalman_Filters.ipynb** contains the experiments we have done with Kalman Filters.

**Demo Videos and Audios**

This folder contains the Demo Videos and Audio recorded for demonstration purpose. 

The rest of the repository contains Input data, Output Results and Analysis for the experiments done to validate our system. Continous Folder contains the experiments done for continous variation of BPM whereas Step Folder contains the experiments pertaining to step transitions. 
