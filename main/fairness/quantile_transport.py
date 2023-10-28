import numpy as np
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF

class EQF:
    def __init__(self, 
                 sample_data,
                 ):
        self._calculate_eqf(sample_data)

    def _calculate_eqf(self,sample_data):
        sorted_data = np.sort(sample_data)
        linspace  = np.linspace(0,1,num=len(sample_data))
        self.interpolater = interp1d(linspace, sorted_data)
        self.min_val = sorted_data[0]
        self.max_val = sorted_data[-1]

    def __call__(self, value_):
        try:
            return self.interpolater(value_)
        except ValueError:
            if value_ < self.min_val:
                return 0.0
            elif value_ > self.max_val:
                return 1.0
            else:
                raise ValueError('Error with input value')
            

class Calibrator:
    def __init__(self):
        self.sens_0 = None
        self.sens_1 = None

    def fit(self,
            scores: np.ndarray, 
            sensitive_vec: np.ndarray):
        val_sens_0, val_sens_1 = set(sensitive_vec)

        self.sens_0 = val_sens_0
        self.sens_1 = val_sens_1

        # Split scores vector
        scores_0 = scores[sensitive_vec == self.sens_0]
        scores_1 = scores[sensitive_vec == self.sens_1]

        self.p0 = len(scores_0) / len(scores)
        self.p1 = len(scores_1) / len(scores)

        # Fit the ecdf and eqf objects
        self.eqf_0 = EQF(scores_0)
        self.eqf_1 = EQF(scores_1)

        self.ecdf_0 = ECDF(scores_0)
        self.ecdf_1 = ECDF(scores_1)

    def transform(self, 
                  scores: np.ndarray, 
                  sensitive_vec: np.ndarray):
        if self.sens_0 is None:
            raise ValueError('to transform the data you must first call the fit() function')
        
        # Split scores but keep indices
        idx_0 = np.where(sensitive_vec == self.sens_0)
        idx_1 = np.where(sensitive_vec == self.sens_1)

        scores_0 = scores[idx_0]
        scores_1 = scores[idx_1]

        new_scores_0 = np.zeros_like(scores_0)
        new_scores_1 = np.zeros_like(scores_1)

        # Add up
        new_scores_0 += self.p0*self.eqf_0(self.ecdf_0(scores_0))
        new_scores_0 += self.p1*self.eqf_1(self.ecdf_0(scores_0))

        new_scores_1 += self.p0*self.eqf_0(self.ecdf_1(scores_1))
        new_scores_1 += self.p1*self.eqf_1(self.ecdf_1(scores_1))
        
        # Recombine
        all_new_scores = np.zeros_like(scores)
        all_new_scores[idx_0] = new_scores_0
        all_new_scores[idx_1] = new_scores_1

        return all_new_scores
    