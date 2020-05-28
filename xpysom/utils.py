import subprocess
import multiprocessing

def find_cuda_cores():
    try:
        
        return int(subprocess.check_output("nvidia-settings -q CUDACores -t", shell=True))
    except:
        print("Could not infer #cuda_cores")
        return 0

def find_cpu_cores():
    try:
        return multiprocessing.cpu_count()
    except:
        print("Could not infer #CPU_cores")
        return 0