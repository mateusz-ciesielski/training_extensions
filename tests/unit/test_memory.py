import psutil
import time


def test_memory_bound():
    print(f"mem info: {psutil.virtual_memory()}")
    alloc = []
    unit_size = 1024 * 1024 * 100
    alloc_unit = " " * unit_size
    idx = 0
    while True:
        alloc.append(alloc_unit + str(idx))
        print(f"mem info({idx}): {psutil.virtual_memory()}")
        idx += 1
        time.sleep(1)
