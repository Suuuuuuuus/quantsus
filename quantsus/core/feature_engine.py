import numpy as np

class SusFeatureEngine:
    def __init__(self, features, feature_names=None, window_size=1):
        """
        features: dict[str, pd.DataFrame]
                  each df shape = (time, assets) or (time,)
        window_size: number of past timesteps to include
        """
        self.features = features
        self.feature_names = feature_names or list(features.keys())
        self.window_size = window_size

        # infer dimensions
        sample_df = features[self.feature_names[0]]
        sample_val = sample_df.iloc[0]

        if np.isscalar(sample_val):
            self.n_assets = 1
        else:
            self.n_assets = len(sample_val)

    def get_single_step(self, t):
        """
        Get feature vector at time t
        shape = (n_features * n_assets,)
        """
        vec = []

        for name in self.feature_names:
            df = self.features[name]
            val = df.iloc[t]

            if np.isscalar(val):
                vec.append(np.array([val]))
            else:
                vec.append(val.values)

        return np.concatenate(vec)

    def get_state(self, t):
        """
        Returns stacked feature vector over window
        shape = (window_size * feature_dim,)
        """
        states = []

        for i in range(self.window_size):
            idx = t - i

            if idx < 0:
                # pad with zeros for early timesteps
                states.append(np.zeros_like(self.get_single_step(0)))
            else:
                states.append(self.get_single_step(idx))

        # stack in time order (oldest → newest)
        states = states[::-1]

        return np.concatenate(states)