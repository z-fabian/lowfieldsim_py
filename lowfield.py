import numpy as np


class InParam:
    def __init__(self,
                 k_high,
                 B_high,
                 B_low,
                 tissue,
                 sequence,
                 TR_high,
                 TR_low,
                 TE_high,
                 TE_low,
                 BW_high,
                 BW_low,
                 theta,
                 n_cov,
                 T1_high=None,
                 T1_low=None,
                 T2=None):
        self.k_high = k_high
        self.B_high = B_high
        self.B_low = B_low
        self.tissue = tissue
        self.sequence = sequence
        self.TR_high = TR_high
        self.TR_low = TR_low
        self.TE_high = TE_high
        self.TE_low = TE_low
        self.BW_high = BW_high
        self.BW_low = BW_low
        self.theta = theta
        self.n_cov = n_cov
        self.T1_high = T1_high
        self.T1_low = T1_low
        self.T2 = T2


def lowfieldgen(in_param, seed=None, only_noise=False):
    # LOWFIELDGEN_TEST simulates low field noise
    # | [k_low] = lowfieldgen_test(in_param)
    # |
    # | Output:
    # | k_low: simulated kspace data at low field in format
    # | [Nkx, Nky, Nkz, Nt, Ncoil]
    # |
    # | Input:
    # | in_param: input parameter structure, details below:
    # |.k_high: kspace data aquired at high field in format
    # | [Nkx Nky Nkz Nt Ncoil]
    # |.B_high(T): B0 field strength at which data was acquired
    # | choose from (0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.5, 3)
    # | or other values(need to specify. T1_high)
    # |.B_low(T): simulated B0 field strength
    # | choose from (0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.5, 3)
    # | or other values(need to specify .T1_low)
    # |.tissue: 'muscle'
    # | 'kidney'
    # | 'white matter'
    # | 'gray matter'
    # | 'liver'
    # | 'fat'
    # | 'other': need to specify .T1_low .T1_high .T2
    # |.sequence: currently support:
    # | 'SpinEcho'
    # | 'GradientEcho'
    # | 'bSSFP'
    # | 'InversionRecovery'
    # |.TR_high(ms): TR of kspace data
    # |.TR_low(ms): TR of simulated low field data
    # |.TE_high(ms): TE of kspace data
    # |.TE_low(ms): TE of simulated low field data
    # |.BW_high(kHz): Readout bandwidth of kspace data
    # |.BW_low(kHz): Readout bandwidth of simulated low field data
    # |.theta(degree): flip angle
    # |.n_cov: noise covariance matrix [Ncoil, Ncoil],
    # | if a single number is entered, a diagonal matrix
    # | will be used
    # | (optional):
    # |.T1_high(s): required only if T1 values of tissue type is
    # | 'other' or B_high is NOT from
    # | (0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.5, 3)
    # |.T1_low(s): required only if T1 values of tissue type is
    # | 'other' or B_low is NOT from
    # | (0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.5, 3)
    # |.T2(s): required only if T2 values if tissue type is
    # | 'other'
    # | seed: (int) random seed to generate noise. If None, a new seed will be generated.
    # | only_noise: (bool) if True, no signal scaling is performed, only additive noise is simulated.
    # (c) written by Ziyue Wu, Feburary 2014.
    # (c) modified by Weiyi Chen, August 2014.
    # Python port by Zalan Fabian, December, 2020
    # University of Southern California
    # https: // mrel.usc.edu
    # T1 & T2 Correction Table REF: Principles of MRI.D.G.Nishimura

    B0_set = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.5, 3]

    tissue_type = ['muscle',
                   'kidney',
                   'white matter',
                   'gray matter',
                   'liver',
                   'fat',
                   'other']
    T1 = {}
    T1['muscle'] = np.array([0.28, 0.37, 0.45, 0.50, 0.55, 0.73, 0.87, 1.42])
    T1['kidney'] = np.array([0.34, 0.39, 0.44, 0.47, 0.50, 0.58, 0.65, 1.19])
    T1['wm'] = np.array([0.31, 0.39, 0.45, 0.49, 0.53, 0.68, 0.78, 1.08])
    T1['gm'] = np.array([0.40, 0.49, 0.55, 0.61, 0.65, 0.82, 0.92, 1.82])
    T1['liver'] = np.array([0.18, 0.23, 0.27, 0.30, 0.32, 0.43, 0.49, 0.81])
    T1['fat'] = np.array([0.17, 0.18, 0.20, 0.21, 0.22, 0.24, 0.26, 0.30])

    T2 = {}
    T2['muscle'] = 0.047
    T2['kidney'] = 0.058
    T2['wm'] = 0.092
    T2['gm'] = 0.1
    T2['liver'] = 0.043
    T2['fat'] = 0.085

    data_dims = in_param.k_high.shape
    assert 3 <= len(data_dims) <= 5
    if len(data_dims) == 5:  # Multi-coil time-series
        Nkx, Nky, Nkz, Nt, Ncoil = data_dims
    elif len(data_dims) == 4:  # Multi-coil data
        Nkx, Nky, Nkz, Ncoil = data_dims
        Nt = 1
    elif len(data_dims) == 3:  # Single-coil data
        Nkx, Nky, Nkz = data_dims
        Nt = 1
        Ncoil = 1

    # ------------------------------------------------------------
    # PASSING PARAMETERS
    # ------------------------------------------------------------
    
    # Covariance matrix
    if isinstance(in_param.n_cov, (list, tuple)):
        n_cov = np.array(in_param.n_cov)
    elif isinstance(in_param.n_cov, np.ndarray):
        assert len(in_param.n_cov.shape) < 3
        if len(in_param.n_cov.shape) == 2:
            n_cov = in_param.n_cov
        else:
            n_cov = in_param.n_cov * np.eye(Ncoil)
    else:
        n_cov = float(in_param.n_cov) * np.eye(Ncoil)

    # T1 at high field
    if in_param.T1_high is None:
        if not ((in_param.B_high in B0_set) and  (in_param.tissue in tissue_type)):
            raise ValueError('Specify T1 value at high field strength.')
        else:
            ind = B0_set.index(in_param.B_high)
            if in_param.tissue is not 'other':
                T1_high = T1[in_param.tissue][ind]
            else:
                raise ValueError('Specify T1 value at high field strength.')
    else:
        T1_high = in_param.T1_high

    if in_param.T1_low is None:
        if not ((in_param.B_low in B0_set) and  (in_param.tissue in tissue_type)):
            raise ValueError('Specify T1 value at low field strength.')
        else:
            ind = B0_set.index(in_param.B_low)
            if in_param.tissue is not 'other':
                T1_low = T1[in_param.tissue][ind]
            else:
                raise ValueError('Specify T1 value at low field strength.')
    else:
        T1_low = in_param.T1_low

    # T2
    if in_param.T2 is None:
        if not (in_param.tissue in tissue_type):
            raise ValueError('Specify T2 value.')
        else:
            if in_param.tissue is not 'other':
                T2 = T2[in_param.tissue]
            else:
                raise ValueError('Specify T2 value.')
    else:
        T2 = in_param.T2

    # TE
    TE_high = in_param.TE_high / 1000  # ms --> sec
    TE_low = in_param.TE_low / 1000  # ms --> sec

    # TR
    TR_high = in_param.TR_high / 1000  # ms --> sec
    TR_low = in_param.TR_low / 1000  # ms --> sec

    # Flip angle
    theta = in_param.theta * np.pi / 180  # degree --> rad

    # Readout bandwidth
    BW_high = in_param.BW_high
    BW_low = in_param.BW_low

    # ------------------------------------------------------------
    # SIGNAL SCALING
    # ------------------------------------------------------------
    if not only_noise:
        E1_h = np.exp(-TR_high / T1_high)
        E1_l = np.exp(-TR_low / T1_low)
        E2_h = np.exp(-TE_high / T2)
        E2_l = np.exp(-TE_low / T2)

        a = in_param.B_low / in_param.B_high

        if in_param.sequence == 'SpinEcho':
            fx = ((1 - E1_l) / (1 - E1_l * np.cos(theta))) \
            / ((1 - E1_h) / (1 - E1_h * np.cos(theta))) \
            *(E2_l / E2_h)
            scaleS = fx * (a ** 2)
        elif in_param.sequence == 'GradientEcho':
            fx = ((1 - E1_l) / (1 - E1_l * np.cos(theta))) \
            / ((1 - E1_h) / (1 - E1_h * np.cos(theta))) \
            *(E2_l / E2_h)
            scaleS = fx * (a ** 2)
        elif in_param.sequence == 'bSSFP':
            raise ValueError('bSSFP not supported')
        elif in_param.sequence == 'InversionRecovery':
            raise ValueError('InversionRecovery not supported')
    else:
        scaleS = 1.0

    # ------------------------------------------------------------
    # NOISE SCALING
    # ------------------------------------------------------------
    b = BW_low / BW_high
    scaleN = np.sqrt(a ** 2 * b - (a ** 4) * (fx ** 2))
    vals, vecs = np.linalg.eig(n_cov)
    d = np.diag(vals)

    # Generate new random seed
    np.random.seed(seed)
    noise = np.random.normal(size=(Ncoil, Nkx * Nky * Nkz * Nt))
    noise = scaleN * np.matmul(vecs, np.matmul(np.sqrt(d), noise))
    noise = np.transpose(noise, (1, 0))
    noise = np.reshape(noise, data_dims)

    # ------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------
    k_low = in_param.k_high * scaleS + noise
    return k_low