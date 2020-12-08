# This code adds noise to the fastMRI dataset to simulate low field acquisition
# Only works for single-coil knee data

# TODO: - multi-coil support
#       - read BW from ISMRM header somehow
#       - brain data support

from lowfield import InParam, lowfieldgen
from utils import ifft2_np
import numpy as np
import h5py
import shutil
import xmltodict
from pathlib import Path
from argparse import ArgumentParser

def estimate_noise_cov(kspace, width=10):
    s, h, w = kspace.shape
    x = np.stack([ifft2_np(kspace[i, ...]) for i in range(s)], axis=0)
    x_re = np.real(x)
    x_im = np.imag(x)
    bg_re = np.concatenate((x_re[:, :, :width], x_re[:, :, -width:]), axis=-1)
    bg_im = np.concatenate((x_im[:, :, :width], x_im[:, :, -width:]), axis=-1)
    bg_all = np.concatenate((bg_re, bg_im), axis=-1)
    noise_std = np.std(bg_all)
    return noise_std**2


def main(args):
    # Main conversion routine
    in_files = list(Path(args.input_path).iterdir())

    for i, in_file in enumerate(in_files):
        print('\rConverting ', i+1, '/', len(in_files), end='')
        # Open next volume
        in_data = h5py.File(in_file, 'r')
        k_high = in_data['kspace']

        # Read parameters from ISMRM header
        xml_header = in_data['ismrmrd_header'][()].decode('UTF-8')
        dict_header = xmltodict.parse(xml_header)['ismrmrdHeader']
        B_high = float(dict_header['acquisitionSystemInformation']['systemFieldStrength_T'])
        B_high = 3.0 if B_high - 2.25 > 0 else 1.5   # Low field sim only supports 1.5T or 3T field strenghts
        TR_high = float(dict_header['sequenceParameters']['TR'])
        TE_high = float(dict_header['sequenceParameters']['TE'])
        theta = float(dict_header['sequenceParameters']['flipAngle_deg'])
        tissue = 'muscle' if args.dataset_type == 'knee' else 'gray matter'  # No support for brain data yet!

        # Estimate noise covariance
        n_cov = estimate_noise_cov(k_high, width=10)

        # Set up input parameters for low field sim
        k_high = np.transpose(k_high, axes=[1, 2, 0])  # Need to be in order [Nkx Nky Nkz] for low field sim
        input_params = InParam(k_high=k_high,
                              B_high=B_high,
                              B_low=args.B_low,
                              tissue=tissue,
                              sequence='SpinEcho',
                              TR_high=TR_high,
                              TR_low=TR_high,
                              TE_high=TE_high,
                              TE_low=TE_high,
                              BW_high=62.5,
                              BW_low=62.5,
                              theta=theta,
                              n_cov=n_cov,
                              T1_high=None,
                              T1_low=None,
                              T2=None
                            )

        # Get simulated low field data
        k_low = lowfieldgen(input_params, seed=None)  # Set seed here to make generated dataset deterministic
        k_low = np.transpose(k_low, axes=[2, 0, 1])  # Permute axes back to original

        # Create output file and write low field data
        in_data.close()
        out_file = str(Path(args.output_path).joinpath('lf_'+str(in_file.name)))
        shutil.copyfile(in_file, out_file)
        out_data = h5py.File(out_file, 'r+')
        kspace = out_data['kspace']
        kspace[...] = k_low
        out_data.close()
    print('\nDone.')


def build_args():
    # ------------------------
    # INPUT ARGUMENTS
    # ------------------------
    parser = ArgumentParser()
    parser.add_argument("--input-path", type=str,
                        help='Path to folder containing high field strength data.')
    parser.add_argument("--output-path", type=str,
                        help='Path to output folder where simulated low field strength data will be saved.')
    parser.add_argument("--B-low", type=float,
                        help='Simulated low field strength [T]. Must be from [0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.5, 3].')
    parser.add_argument("--dataset-type", type=str, default='knee',
                        help='Choose fastMRI dataset from ["knee", "brain"] . Currently only supports knee.')
    args = parser.parse_args()
    return args


def run_cli():
    args = build_args()

    # ---------------------
    # CONVERT DATASET
    # ---------------------
    main(args)


if __name__ == "__main__":
    run_cli()

