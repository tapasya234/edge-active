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

    def state_dict(self):
        """Returns the state of the stopper as a dictionary."""
        return {
            "patience": self.patience,
            "counter": self.counter,
            "delta": self.delta,
            "best_value": self.best_value,
            "mode": self.mode,
        }

    def load_state_dict(self, state_dict):
        """Loads the stopper state from a dictionary."""
        self.patience = state_dict["patience"]
        self.counter = state_dict["counter"]
        self.delta = state_dict["delta"]
        self.best_value = state_dict["best_value"]
        self.mode = state_dict["mode"]
