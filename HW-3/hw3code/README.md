## Set up environment
We recommend you to set up a conda environment for packages used in this homework.

For part II, you need access to GPUs. 

You can either use HPC server (see documentation here[https://github.com/Athul-R/NYU-HPC-Tutorials]) or use API services like lightning.ai. If using such service, you can skip conda commands and just install requirements in your studio.
```
conda create -n 2590-hw3 python=3.12.11
conda activate 2590-hw3
pip install -r requirements.txt
```

To register a jupyter kernel from your conda environment, use
```
conda activate 2590-hw3
python3 -m ipykernel install --user --name envkernel
```
