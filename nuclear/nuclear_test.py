import time 
import sys
import numpy as np
import random
import cudaq
from cudaq import spin
from typing import List, Tuple


if cudaq.num_available_gpus() == 0:
    print("This example requires a GPU to run. No GPU detected.")

cudaq.set_target("nvidia", option="mqpu")
#cudaq.set_target('nvidia', option='fp64')
cudaq.mpi.initialize()

num_ranks = cudaq.mpi.num_ranks() 
rank = cudaq.mpi.rank() 

#print('rank', rank, 'num_ranks', num_ranks)

if rank == 0:     
    print('mpi is initialized? ', cudaq.mpi.is_initialized())

qubit_count = 1 
num_pairs = 1

def get_H(one_body_couplings: np.ndarray,two_body_couplings:np.ndarray):
    '''
        one_body_couplings: 1D array of one body couplings
        two_body_couplings: 2D array of two body couplings
        Generate the Pauli string Hamiltonian given the one body and two body couplings
    '''
    n=len(one_body_couplings)
    assert two_body_couplings.shape[0]==n and two_body_couplings.shape[1]==n
    H1 = 0*spin.i(0)
    H2 = 0*spin.i(0)*spin.i(0)
    
    for i in range(n):
        H1 += (one_body_couplings[i]+two_body_couplings[i,i])*spin.z(i)
    
    for i in range(n):
        for j in range(i+1,n):
            H2 += two_body_couplings[i,j]*(spin.x(i)*spin.x(j)+spin.y(i)*spin.y(j))
    return H1+H2


def get_Hsq(one_body_couplings: list[float],two_body_couplings: list[float]):
    '''
        one_body_couplings: 1D array of one body couplings
        two_body_couplings: 2D array of two body couplings
        Generate the Pauli string of the square of  the Hamiltonian given the one body and two body couplings
    '''
    h=get_H(one_body_couplings, two_body_couplings)
    return h*h

def get_N(n): 
    '''
        n: number of qubits
        Generate the Pauli string for the total particle number
    '''
    op = 0*spin.i(0)

    for i in range(n):
        op += spin.i(i)-spin.z(i)
    return op

def get_Nsq(n):
    '''
        n: number of qubits
        Generate the Pauli string for the total particle number squared
    '''
    nop=get_N(n)
    return nop*nop

@cudaq.kernel # This is the custom VQE anstaz from the previous dicsussion
def AGP_VQE_ansatz(thetas: List[float], qubit_count: int, num_pairs: int):
    '''
        thetas: list of parameters for ansatz
        qubit_count: number of qubits
        num_pairs: number of pairs
        AGP CUDA-Q kernel.
    '''
    # Next, we can allocate the qubits to the kernel via `qvector()`.
    n = len(thetas)+1
    qubits = cudaq.qvector(n)

    # Now we can begin adding instructions to apply to thess qubits.
    x(qubits[0])
    ry(thetas[0],qubits[1])
    x.ctrl(qubits[1], qubits[0])
    for i in range(1,n-1):
        ry.ctrl(thetas[i],qubits[i],qubits[i+1])
        x.ctrl(qubits[i+1], qubits[i])
        
@cudaq.kernel
def UCCSD_VQE_ansatz(thetas: list[float], qubit_count: int, num_pairs: int):
    '''
        thetas: list of parameters for ansatz
        qubit_count: number of qubits
        num_pairs: number of pairs
        UCCSd CUDA-Q kernel.
    '''
    
    qubits = cudaq.qvector(qubit_count)

    for i in range(num_pairs):
        x(qubits[i])
    cudaq.kernels.uccsd(qubits, thetas, num_pairs, qubit_count)

        
def objective_function(parameter_vector: List[float], 
                       hamiltonian: cudaq.SpinOperator, 
                       gradient_strategy: cudaq.gradients, 
                       kernel: cudaq.kernel,
                       qubit_count: int,
                       num_pairs: int,  
                       verbose: bool,
                       execution=None) -> Tuple[float, List[float]]:
    """
        parameter_vector: list of parameters for ansatz
        hamiltonian: Hamiltonian of the system 
        gradient_strategy: how gradients are computed
        qubit_count: number of qubits
        num_pairs: number of pairs
        
        Objective function returns cost and gradient vector
    """

    # Call `cudaq.observe` on the spin operator and ansatz at the
    # optimizer provided parameters. This will allow us to easily
    # extract the expectation value of the entire system in the
    # z-basis.

    # We define the call to `cudaq.observe` here as a lambda to
    # allow it to be passed into the gradient strategy as a
    # function. 
    get_result = lambda parameter_vector: cudaq.observe(
        kernel, hamiltonian, parameter_vector, qubit_count, num_pairs, execution = execution).expectation()
    # `cudaq.observe` returns a `cudaq.ObserveResult` that holds the
    # counts dictionary and the `expectation`.
    cost = get_result(parameter_vector)
    if verbose:
        print(f"<H> = {cost}")
    # Compute the gradient vector using `cudaq.gradients.STRATEGY.compute()`.
    gradient_vector = gradient_strategy.compute(parameter_vector, get_result,
                                                cost)

    # Return the (cost, gradient_vector) tuple.
    return cost, gradient_vector


# Compute the energy and occupation number for various optimizers 
def compare_optimizers(H: cudaq.SpinOperator,
                       Hsq: cudaq.SpinOperator,
                       Nop: cudaq.SpinOperator,
                       Nopsq: cudaq.SpinOperator,
                       ansatz: cudaq.kernel,
                       qubit_count: int,
                       num_pairs: int,
                       gradient: cudaq.gradients,
                       optimizer_list: list[cudaq.optimizers],
                       verbose=False,
                       execution=None):
    '''
        H: Hamiltonian spin operator,
        Hsq: Hamiltonian squared spin operator,
        Nop: Number spin operator,
        Nopsq: Number squared spin operator,
        ansatz: VQE ansatz,
        qubit_count: number of qubits,
        num_pairs: number of pairs,
        gradient: gradient strategy,
        optimizer_list: list of optimizers
    
        returns energies, occupation numbers, parameters for the ansatz for the given list of optmizers
    '''
    energies=[]
    numbers=[]
    params=[]
    for i,optimizer in enumerate(optimizer_list):
        obj_func = lambda x: objective_function(x,hamiltonian=H,gradient_strategy=gradient,\
                                            kernel=ansatz,qubit_count=qubit_count,num_pairs=num_pairs\
                                                ,verbose=verbose,execution=execution)
        # get the optimal parameters
        energy, parameter = optimizer.optimize(dimensions=qubit_count, function=obj_func)
        params.append(parameter)
        n=cudaq.observe(ansatz,Nop,parameter,qubit_count,num_pairs,execution=execution).expectation()
        nsq=cudaq.observe(ansatz,Nopsq,parameter,qubit_count,num_pairs,execution=execution).expectation()
        energysq=cudaq.observe(ansatz,Hsq,parameter,qubit_count,num_pairs,execution=execution).expectation()
        energies.append([energy,np.sqrt(abs(energy**2-energysq))])
        numbers.append([n,np.sqrt(abs(n**2-nsq))])
    
    return energies,numbers,params

def get_setup(N: int):
    '''
        N: sytem size
    '''
    global qubit_count, num_pairs
    qubit_count=N
    num_pairs = 1

    parameter_count=cudaq.kernels.uccsd_num_parameters(num_pairs,qubit_count)

    # Pick a sequence of couplings for the one body term
    obc = np.array(np.arange(N)[::-1])*(-1)
    
    # Generate a random symmetric real matric for the two body term
    tbc = np.random.rand(N,N)
    tbc += tbc.T -1

    # generate the operators
    hamiltonian = get_H(obc,tbc)
    hamiltonian_sq = get_Hsq(obc,tbc)
    number = get_N(N)
    number_sq = get_Nsq(N)
    return qubit_count,num_pairs,parameter_count,hamiltonian,hamiltonian_sq,number,number_sq    
    
    
    
def main():
    # Pick a system size
    # N = 6  -> (J_max=0,J_max=3/2,J_max=5/2) for isotope Oxygen-18
    
    N = 12
    
    qubit_count,num_pairs,parameter_count,hamiltonian,hamiltonian_sq,number,number_sq = get_setup(N)

    gradient = cudaq.gradients.CentralDifference()

    cudaq.set_random_seed(11)  # make repeatable

    
    # define a list of optimizers so we can compare 
    optimizer_list=[cudaq.optimizers.Adam(),cudaq.optimizers.COBYLA(),cudaq.optimizers.LBFGS()]#cudaq.optimizers.NelderMead(),
    optimizer_names=['Adam','COBYLA','LBFGS']#'NelderMead'
    # Single node, single GPU.
    t = time.time()
    
    energies_uccsd,numbers_uccsd,params_uccsd=compare_optimizers(hamiltonian,hamiltonian_sq,number,number_sq,\
                       UCCSD_VQE_ansatz,qubit_count,num_pairs,gradient,optimizer_list,verbose=False)
    #sys.exit()
    
    energies_agp,numbers_agp,params_agp=compare_optimizers(hamiltonian,hamiltonian_sq,number,number_sq,\
                       AGP_VQE_ansatz,qubit_count,num_pairs,gradient,optimizer_list,verbose=False)
    
    t0 = time.time() - t 
    
    agp_index =np.argmin(np.array(energies_agp)[:,0])
    energy_agp_single_gpu = energies_agp[agp_index]
    number_agp_single_gpu = numbers_agp[agp_index]
    param_agp_single_gpu = params_agp[agp_index]
    
    uccsd_index =np.argmin(np.array(energies_uccsd)[:,0])
    energy_uccsd_single_gpu = energies_agp[uccsd_index]
    number_uccsd_single_gpu = numbers_uccsd[uccsd_index]
    param_uccsd_single_gpu = params_uccsd[uccsd_index]
    
    
    
    # If we have multiple GPUs/ QPUs available, we can parallelize the workflow with the addition of an argument in the observe call.

    # Single node, multi-GPU.
    t = time.time()
    energies_uccsd,numbers_uccsd,params_uccsd=compare_optimizers(hamiltonian,hamiltonian_sq,number,number_sq,\
                       UCCSD_VQE_ansatz,qubit_count,num_pairs,gradient,optimizer_list,verbose=False,execution=cudaq.parallel.thread)
    
    energies_agp,numbers_agp,params_agp=compare_optimizers(hamiltonian,hamiltonian_sq,number,number_sq,\
                       AGP_VQE_ansatz,qubit_count,num_pairs,gradient,optimizer_list,verbose=False,execution=cudaq.parallel.thread)
    
    t1 = time.time() - t 
    
    agp_index =np.argmin(np.array(energies_agp)[:,0])
    energy_agp_parallel_thread = energies_agp[agp_index]
    number_agp_parallel_thread = numbers_agp[agp_index]
    param_agp_parallel_thread = params_agp[agp_index]
    
    uccsd_index =np.argmin(np.array(energies_uccsd)[:,0])
    energy_uccsd_parallel_thread = energies_agp[uccsd_index]
    number_uccsd_parallel_thread = numbers_uccsd[uccsd_index]
    param_uccsd_parallel_thread = params_uccsd[uccsd_index]
    
    
    # Multi-node, multi-GPU.
    t = time.time()
    energies_uccsd,numbers_uccsd,params_uccsd=compare_optimizers(hamiltonian,hamiltonian_sq,number,number_sq,\
                       UCCSD_VQE_ansatz,qubit_count,num_pairs,gradient,optimizer_list,verbose=False,execution=cudaq.parallel.mpi)
    
    energies_agp,numbers_agp,params_agp=compare_optimizers(hamiltonian,hamiltonian_sq,number,number_sq,\
                       AGP_VQE_ansatz,qubit_count,num_pairs,gradient,optimizer_list,verbose=False,execution=cudaq.parallel.mpi)
    
    t2 = time.time() - t 
    
    agp_index =np.argmin(np.array(energies_agp)[:,0])
    energy_agp_mpi = energies_agp[agp_index]
    number_agp_mpi = numbers_agp[agp_index]
    param_agp_mpi = params_agp[agp_index]
    
    uccsd_index =np.argmin(np.array(energies_uccsd)[:,0])
    energy_uccsd_mpi = energies_agp[uccsd_index]
    number_uccsd_mpi = numbers_uccsd[uccsd_index]
    param_uccsd_mpi = params_uccsd[uccsd_index]

    if rank == 0: 
    
        print(f'single gpu, Energy AGP = {energy_agp_single_gpu[0]}+/-{energy_agp_single_gpu[1]};  Energy UCCSD = {energy_uccsd_single_gpu[0]}+/-{energy_uccsd_single_gpu[1]}; \n Number AGP = {number_agp_single_gpu[0]}+/-{energy_agp_single_gpu[1]};  Number UCCSD = {energy_uccsd_single_gpu[0]}+/-{energy_uccsd_single_gpu[1]}; time = {t0}')
        print("")
        print(f'single node multi gpu, Energy AGP = {energy_agp_parallel_thread[0]}+/-{energy_agp_parallel_thread[1]};  Energy UCCSD = {energy_uccsd_parallel_thread[0]}+/-{energy_uccsd_parallel_thread[1]}; time = {t1}')
        print("")
        print(f'multi node multi gpu, Energy AGP = {energy_agp_mpi[0]}+/-{energy_agp_mpi[1]};  Energy UCCSD = {energy_uccsd_mpi[0]}+/-{energy_uccsd_mpi[1]}; time = {t2}')
    

main()

cudaq.mpi.finalize()