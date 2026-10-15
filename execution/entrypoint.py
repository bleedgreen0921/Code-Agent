"""PID 1 watchdog: the absolute Run deadline survives loss of every Worker."""

import os
import signal
import sys
import time

deadline = float(sys.argv[1])


def expire(*_):
    # Exiting PID 1 causes the kernel to terminate the entire PID namespace.
    os._exit(124)


signal.signal(signal.SIGTERM, expire)
signal.signal(signal.SIGINT, expire)
while True:
    remaining = deadline - time.time()
    if remaining <= 0:
        expire()
    time.sleep(min(remaining, 1))
