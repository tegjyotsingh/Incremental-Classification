Incremental-Classification
==========================

Particle Filtering Markov Chain methodology for Incremental classification of Streaming data
--------------------------------------------------------------------------------------------

This project aims at developing a model for Incremental classification of Streaming Data that exhibit concept drift. The model is
based on the Particle Filtering methodology. The project is implemented using Python. MOA from Weka was used for performing experimentation
on the existing methiodologies for performing a performance comparison. 

The report was presented for the Computational Cognitive Sciences Class at UofL.


The folder is organized as follows:

> dataset: Contains the three datasets used for evaluation. 
           *SEA data stream: Synthetic dataset with 60,000 samples
           *TJSS Stream: Synthetic stream developed with various drifts(See report for details)
           *EM stream: Real wordl stream with ~45,000 samples and 7 attributes

> experimentation: Experimentation files with the JSON files from results of the python scripts. 
                   Results obtained from MOA have a .pred extension and that from python have .dat extension. 
                   Convert_json.py in appropritate folder performs conversion from .pred to .dat.
                   
                   The plots folder has the scripts to obtain appropriate plots from these JSON files. 
                   The script written in python can be configured for different chunk sizes and datasets to 
                   suit the comparative needs.

> scripts: Contains the scrap scripts written for internmediate operations and also the final deliverable scipts in the model folder
           Scripts for JSON  conversion, plotting and other presentation taksa re located in the appropriate experimentation subfolder
           

> report: The draft report of the model presented for the purposes of the class. Written in IEEE Conference style.  
