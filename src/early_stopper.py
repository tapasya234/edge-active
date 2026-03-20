class EarlyStopper:
    def __init__(self, patience=2, delta=0.25, mode="max"):
        self.patience = patience
        self.counter = 0
        self.delta = delta
        self.best_value = None
        self.mode = mode

    def check(self, metric):
        if self.best_value == None:
            self.best_value = metric
            return False

        if self.mode == "max":
            if metric > self.best_value + self.delta:
                self.best_value = metric
                self.counter = 0
                return False

            self.counter += 1
            return self.counter >= self.patience

        if metric < self.best_value - self.delta:
            self.best_value = metric
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience
