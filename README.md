# MAPPER 2.0

This repository contains the source code for **MAPPER 2.0**, that was used for quantitative analysis of *Drosophila* wings for the manuscript **"A machine-learning enhanced data fusion pipeline identifies new G-protein-based regulators of *Drosophila* wing growth and morphogenesis."** MAPPER 2.0 provides a graphical user interface (GUI) for segmentation, feature extraction, and batch processing of adult *Drosophila* wings. Additional details on the workflow and GUI usage are provided in the documentation.

## Installation

### Prerequisites
Before installing MAPPER 2.0, please install **Anaconda** on your computer from the official Anaconda website.

### Setup Instructions

1. Open **Anaconda Prompt**.

2. Navigate to the folder where you downloaded the MAPPER 2.0 code using the `cd` command. For example:

        cd Desktop
        cd mapper-app

   You can use the following command to view the contents of your current directory:

        dir

3. Make sure you are inside the **root code directory**, where the configuration file `mapper_env.yml` is located.

4. Create a new conda environment by running:

        conda env create -f mapper_env.yml -n mapper_2_env

   This step may take several minutes because the required packages will be downloaded and installed.

5. Once installation is complete, activate the environment:

        conda activate mapper_2_env

6. Run MAPPER 2.0 by executing:

        python pyMAPPER-v1.py

7. The **MAPPER 2.0 graphical user interface (GUI)** will launch automatically.