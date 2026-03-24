import time

class SusTimer:
    """
    Simple timer class for measuring elapsed time.
    """

    def __init__(self):
        self.start_time = None
        self.elapsed = 0

    def start(self):
        """
        Start or restart the timer.
        """
        self.start_time = time.time()
        self.elapsed = 0

    def stop(self):
        """
        Stop the timer and update elapsed time.
        """
        if self.start_time is None:
            raise RuntimeError("Timer not started.")
        self.elapsed = time.time() - self.start_time
        self.start_time = None
        return self.elapsed

    def elapsed_time(self):
        """
        Return elapsed time without stopping the timer.
        """
        if self.start_time is None:
            return self.elapsed
        return time.time() - self.start_time

    def __enter__(self):
        """
        Support for 'with' context manager.
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()