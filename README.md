# sc24-quantum-tutorial
Repository contains all the tutorial materials for the SC24 workshop [Accelerated Quantum Supercomputing: A Step-by-Step Guide to Expanding Simulation Capabilities and Enabling Interoperability of Quantum Hardware](https://sc24.conference-program.com/presentation/?id=tut167&sess=sess407), a collaboration with NERSC, NVIDIA, and QuEra. 

## Description
GPU-accelerated quantum simulations are increasingly being adopted in hybrid quantum-classical algorithm development to speed up algorithm run-time, to test and implement future parallel QPU workflows, to scale up the size of quantum research, and to deploy workflows where QPUs and GPUs are tightly coupled. This tutorial guides attendees through examples simulated on their laptops to GPUs on NVIDIA Quantum Cloud. We then focus on running industry-relevant quantum research problems on HPC systems. The tutorial begins with an interactive Jupyter notebook demonstrating parallel quantum simulation using open-source CUDA-Q (introductory material). Next, the tutorial enables attendees to deploy quantum software on large scale HPC clusters like Perlmutter to run, for example, a 30,000 term Hamiltonian using 100 GPUs across multiple nodes (intermediate and advanced material). The tutorial ends with a presentation on QuEra machines and their capabilities along with a hands-on example setting up a Quantum Reservoir Models on QuEra’s platform (intermediate and advanced material). This is the software to be used: [https://nvidia.github.io/cuda-quantum/latest/index.html](https://nvidia.github.io/cuda-quantum/latest/index.html)This is the Docker image to be used (or extended to be optimal on Perlmutter): [https://catalog.ngc.nvidia.com/orgs/nvidia/teams/quantum/containers/cuda-quantum](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/quantum/containers/cuda-quantum).

## Tutorial Schedule

The tutorial runs from 8:30am - 5:00pm EST.  There are four sessions separated by breaks.  The tentative agenda for each session follows:

* **8:30-10:00am:** Overview of methods of accelerating quantum simulation with GPUs including a hands-on QAOA example with CUDA-Q 
* 10:00-10:30: Break
* **10:30-noon:** Live demo local install of CUDA-Q followed by an introduction to large scale clusters, how to navigate and use them 
* noon-1:30pm: Lunch break
* **1:30-3:00pm:** Hands-on Example: Quantum Chemistry at NERSC and a industry use case of simulating Hamiltonians of molecules with 30,000 terms
* 3:00-3:30pm: Break 
* **3:30-5:00pm:** Finish the industry use case example, run a Quantum Resevoir Computing example with QuEra, and conclude the session

## Resources
The slides for all the sessions are collated in the file [quantum-accelerated-supercomputing-sc24.pdf](quantum-accelerated-supercomputing-sc24.pdf). Tutorial notebooks and other resources for each session are found in the directories of this repository.
