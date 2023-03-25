import numpy as np
import time

try:
    import cupy as cp
    start_event = cp.cuda.stream.Event()
    stop_event = cp.cuda.stream.Event()

    def time_cp(f, *args, **kwargs):
        start_event.record()
        out = f(*args, **kwargs)
        stop_event.record()
        stop_event.synchronize()
        dt = cp.cuda.get_elapsed_time(start_event, stop_event)/1000
        return dt, out

except Exception as e:
    cp = None
    print("Can't import cupy", e)

def time_np(f, *args, **kwargs):
    t_start = time.perf_counter()
    out = f(*args, **kwargs)
    t_end = time.perf_counter()
    stop_event.synchronize()
    dt = t_end - t_start
    return dt, out

units = {"nsec": 1e-9, "usec": 1e-6, "msec": 1e-3, "sec": 1.0}
precision = 3
def format_time(dt):
    """From timeit"""
    scales = [(scale, unit) for unit, scale in units.items()]
    scales.sort(reverse=True)
    for scale, unit in scales:
        if dt >= scale:
            break

    return "%.*g %s" % (precision, dt / scale, unit)

def generate_randf(*shapes, xp=np):
    res = []
    kwargs = {}
    if xp.__name__ == 'cupy':
        kwargs['dtype'] = xp.float32
    for shape in shapes:
        res.append(xp.random.rand(*shape, **kwargs))
    return res

def generate_randi(*shapes, low=None, high=None, xp=np):
    res = []
    for shape in shapes:
        res.append(xp.random.randint(low, high, shape))
    return res

def bench(f, *input_shapes, generator=generate_randf, kwargs={}, name='', xp=np, repeat=10, warmup=0, dump=False):
    results = []
    for i in range(warmup+repeat):
        args = generator(*input_shapes, xp=xp)
        if xp.__name__ == 'cupy':
            dt, _out = time_cp(f, *args, **kwargs)
        else:
            dt, _out = time_np(f, *args, **kwargs)

        if i >= warmup:
            results.append(dt)

    mean = np.mean(results)
    std = np.std(results)
    max_ = max(results)
    min_ = min(results)

    if dump:
        print(results)

    print(f"{format_time(mean)} ± {format_time(std)} "
        f"per run (mean ± std. dev. of {repeat} runs). "
        f"Max: {format_time(max_)}, min: {format_time(min_)}")
