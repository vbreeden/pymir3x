"""
SpectralFlux.py
Compute the spectral flux between consecutive spectra
This technique can be for onset detection

rectify - only return positive values
Ported from https://github.com/jsawruk/pymir: 30 August 2017
"""


def spectralFlux(spectra, rectify=False):
    spectral_flux = []

    # Compute flux for zeroth spectrum
    flux = 0
    for nth_bin in spectra[0]:
        flux = flux + abs(nth_bin)

        spectral_flux.append(flux)

    # Compute flux for subsequent spectra
    for s in range(1, len(spectra)):
        prev_spectrum = spectra[s - 1]
        spectrum = spectra[s]

        flux = 0
        for mth_bin in range(0, len(spectrum)):
            diff = abs(spectrum[mth_bin]) - abs(prev_spectrum[mth_bin])

            # If rectify is specified, only return positive values
            if rectify and diff < 0:
                diff = 0

            flux = flux + diff

            spectral_flux.append(flux)

    return spectral_flux
