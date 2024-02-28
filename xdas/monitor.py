from time import perf_counter

from tqdm import tqdm


class Monitor:
    def __init__(self, total=None, smoothing=0.3):
        self.pbar = tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024)
        self.smoothing = smoothing
        self.time = {}
        self.iter = {}
        self.ema = {}
        self.cum = {}

    def tic(self, key):
        self.time[key] = perf_counter()

    def toc(self, nbytes):
        time = perf_counter()
        values = list(self.time.values()) + [time]
        for idx, key in enumerate(self.time):
            self.iter[key] = values[idx + 1] - values[idx]
        for key in self.iter:
            if key in self.ema:
                self.ema[key] += self.smoothing * (self.iter[key] - self.ema[key])
            else:
                self.ema[key] = self.iter[key]
        for key in self.iter:
            if key in self.cum:
                self.cum[key] += self.iter[key]
            else:
                self.cum[key] = self.iter[key]
        self.pbar.update(nbytes)
        self.pbar.set_postfix(self.usage_str())

    def close(self):
        self.pbar.set_postfix(self.average_usage_str())
        self.pbar.close()

    def usage(self):
        total = sum(self.iter.values())
        return {key: self.iter[key] / total for key in self.iter}

    def average_usage(self):
        total = sum(self.cum.values())
        return {key: self.cum[key] / total for key in self.cum}

    @staticmethod
    def format(x):
        return f"{100 * x:.1f}%"

    def usage_str(self):
        d = self.usage()
        return {key: self.format(value) for key, value in d.items()}

    def average_usage_str(self):
        d = self.average_usage()
        return {key: self.format(value) for key, value in d.items()}
