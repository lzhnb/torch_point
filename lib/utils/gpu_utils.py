from gpustat import *

def get_free_gpu_process():
    gpu_stat     = GPUStatCollection.new_query()
    gpu_free_idx = []
    for i in range(len(gpu_stat)):
        pids = gpu_stat[i].processes
        if len(pids) == 0: 
            gpu_free_idx.append(i)
            print('gpu[{}]: {}MB with no python process'.format(i, gpu_stat[i].memory_free))
            continue
        python_on = False
        for pid in pids:
            if pid['command'] == 'python': 
                python_on = True
                break
        if not python_on:
            gpu_free_idx.append(i)
            print('gpu[{}]: {}MB with no python process'.format(i,gpu_stat[i].memory_free))
    return gpu_free_idx

def get_free_gpu_memory(free_memory):
    gpu_stat     = GPUStatCollection.new_query()
    gpu_free_idx = []
    for i in range(len(gpu_stat)):
        if gpu_stat[i].memory_free > free_memory:
            gpu_free_idx.append(i)
            print('gpu[{}]: {}MB'.format(i,gpu_stat[i].memory_free))
    return gpu_free_idx

def get_free_gpu(free_memory, mode='memory'):
    return get_free_gpu_process() if mode is not 'memory' \
        else get_free_gpu_memory(free_memory) 


def supervise_gpu(free_memory=10000, nb_gpu=1, mode='memory'):
    import time
    gpu_free_idx = []
    while len(gpu_free_idx) < nb_gpu:
        time.sleep(3)
        gpu_free_idx = get_free_gpu(free_memory, mode)
    return gpu_free_idx[:nb_gpu]