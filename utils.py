import numpy as np
from scipy.linalg import sqrtm


def calc_fid_score(mean_0, cov_0, mean_1, cov_1):
    mean_dist = np.linalg.norm(mean_0 - mean_1) ** 2
    cov_mean = sqrtm(cov_0 @ cov_1)
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    cov_dist = np.trace(cov_0 + cov_1 - 2 * cov_mean)
    fid_score = mean_dist + cov_dist

    return fid_score


if __name__ == "__main__":
    mean_0 = np.random.normal(size=(2048, ))
    mean_1 = np.random.normal(size=(2048, ))
    cov_0 = np.random.normal(size=(2048, 2048))
    cov_0 = cov_0.T @ cov_0
    cov_1 = np.random.normal(size=(2048, 2048))
    cov_1 = cov_1.T @ cov_1
    fid_score = calc_fid_score(mean_0, cov_0, mean_1, cov_1)
    print(fid_score)
