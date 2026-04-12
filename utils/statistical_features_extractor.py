import numpy as np
import warnings
from scipy import stats
from scipy.stats import entropy
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

class StatisticalFeatureExtractor:
    def __init__(self):
        # Pre-compute ASCII character range once
        self.ascii_chars = [chr(i) for i in range(33, 127)]
        self.ascii_set = set(self.ascii_chars)
        self.num_chars = len(self.ascii_chars)
        self.char_to_idx = {char: idx for idx, char in enumerate(self.ascii_chars)}  # ← add this

    def get_character_distributions_vectorized(self, data_cells):
        n_cells = len(data_cells)
        char_matrix = np.zeros((n_cells, self.num_chars), dtype=np.float32)

        for cell_idx, cell in enumerate(data_cells):
            # Filter to ASCII range, convert to indices in one pass
            indices = [self.char_to_idx[c] for c in str(cell) if c in self.char_to_idx]
            if indices:
                np.add.at(char_matrix[cell_idx], indices, 1)

        row_sums = char_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return char_matrix / row_sums

    def generate_graph(self, label, char_distributions, mean_char_dist):
        x = np.arange(char_distributions.shape[1])
        x_smooth = np.linspace(x.min(), x.max(), 100)
        n = min(len(char_distributions), 500)

        plt.figure(figsize=(8, 6))
        for i, row in enumerate(char_distributions[:n]):
            spl = make_interp_spline(x, row, k=3)
            y_smooth = spl(x_smooth)
            y_positive = np.clip(y_smooth, a_min=0, a_max=None)

            plt.plot(x_smooth, y_positive, color='lightblue', alpha=0.2)

        spl = make_interp_spline(x, mean_char_dist, k=3)
        power_smooth = spl(x_smooth)
        y_positive = np.clip(power_smooth, a_min=0, a_max=None)
        plt.plot(x_smooth, y_positive, color='darkblue', alpha=0.7)

        plt.xlabel('ASCII values')
        plt.ylabel('Frequency')
        plt.title(f'Plotting Character Distributions for {label} column type')
        plt.grid(True)
        plt.show()

    def extract_statistical_features_from_column(self, column_data, label, generate_plot=False):
        """
        Extract statistical features from a column.
        """
        # Filter out NaN values once
        valid_data = column_data.dropna()

        if len(valid_data) == 0:
            return self._get_empty_features()

        # Convert to list for faster iteration
        data_list = [str(cell) for cell in valid_data.tolist()]

        # Vectorized length calculation
        value_lengths = np.array([len(cell) for cell in data_list], dtype=np.int32)

        # Vectorized character distribution
        char_distributions = self.get_character_distributions_vectorized(data_list)

        # Calculate mean character distribution
        mean_char_dist = char_distributions.mean(axis=0)

        if generate_plot:
            self.generate_graph(label, char_distributions, mean_char_dist)

        # Build feature dictionary
        statistical_features = {
            char: mean_char_dist[idx]
            for idx, char in enumerate(self.ascii_chars)
        }

        # Length statistics
        statistical_features['min_length'] = value_lengths.min()
        statistical_features['max_length'] = value_lengths.max()
        statistical_features['mean_length'] = value_lengths.mean()
        statistical_features['median_length'] = np.median(value_lengths)

        # Skewness and kurtosis with error handling
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                statistical_features['skew_length'] = stats.skew(value_lengths)
            except:
                statistical_features['skew_length'] = 0.0

            try:
                statistical_features['kurtosis_length'] = stats.kurtosis(
                    value_lengths, fisher=False
                )
            except:
                statistical_features['kurtosis_length'] = -3.0

        # Calculate entropy
        value_counts = valid_data.value_counts()
        probabilities = value_counts / len(valid_data)
        statistical_features['entropy'] = entropy(probabilities, base=2)

        return statistical_features

    def _get_empty_features(self):
        """Return default features for empty columns."""
        features = {char: 0.0 for char in self.ascii_chars}
        features.update({
            'min_length': 0,
            'max_length': 0,
            'mean_length': 0.0,
            'median_length': 0.0,
            'skew_length': 0.0,
            'kurtosis_length': -3.0,
            'entropy': 0.0
        })
        return features
