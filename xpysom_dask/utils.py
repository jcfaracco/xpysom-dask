import subprocess
import multiprocessing

def find_max_cuda_threads():
    try:
        import cupy as cp
        dev = cp.cuda.Device()
        n_smp = dev.attributes['MultiProcessorCount']
        max_thread_per_smp = dev.attributes['MaxThreadsPerMultiProcessor']
        return n_smp*max_thread_per_smp
    except:
        print("Cupy is not available.")
        return 0

def find_cpu_cores():
    try:
        return multiprocessing.cpu_count()
    except:
        print("Could not infer #CPU_cores")
        return 0