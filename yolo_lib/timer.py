from .cfg import USE_GPU
import torch
import time


"""
Wraps cuda event timers and python's perf_counter timers for an abstract timing interface
that supports measuring time on both CPU and GPU
"""


class Timer:
    def __init__(self):
        if USE_GPU:
            self.cuda_event = torch.cuda.Event(enable_timing=True)

    def record(self):
        if USE_GPU:
            self.cuda_event.record()
        else:
            self.cpu_timer = time.perf_counter()

    @staticmethod
    def sync():
        if USE_GPU:
            torch.cuda.synchronize()

    @staticmethod
    def elapsed_seconds(start_timer, end_timer):
        if USE_GPU:
            elapsed_milliseconds = start_timer.cuda_event.elapsed_time(end_timer.cuda_event)
            return elapsed_milliseconds / 1000
        else:
            return end_timer.cpu_timer - start_timer.cpu_timer

