###############################################################################
# Copyright 2024 Ferenc Csikor
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
###############################################################################
import numpy as np
from scipy import ndimage


def offdiagonal(a):
    "Returns the off-diagonal elements of array a as a 1D array."
    return np.concatenate((np.triu(a, k=1), np.tril(a, k=-1)), axis=None)


def calc_center_of_mass(img):
    "Returns the center of mass of the square of image img."
    if img.ndim == 1:
        xdim = int(np.sqrt(img.size))
        img = img.reshape((xdim, xdim))

    return np.array(ndimage.measurements.center_of_mass(np.square(img)))


def calc_wave_vector(img):
    "Returns the dominant wave vector of the power spectrum of image img."
    if img.ndim == 1:
        xdim = int(np.sqrt(img.size))
        img = img.reshape((xdim, xdim))
    else:
        xdim = img.shape[0]

    power_spectr = np.square(np.abs(np.fft.fftshift(np.fft.fft2(img))))
    return (2.0 * np.pi
            * (np.array(ndimage.maximum_position(power_spectr)) - xdim/2)
            / xdim)
