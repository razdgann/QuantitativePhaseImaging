import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift


class QuantitativePhaseImaging:
    def __init__(self, wavelength, distance, pixel_size):
        """
        Initialize the QPI class with fundamental parameters.

        :param wavelength: Wavelength of the light (in meters)
        :param distance: Propagation distance (in meters)
        :param pixel_size: Pixel size of the imaging system (in meters)
        """
        self.wavelength = wavelength
        self.distance = distance
        self.pixel_size = pixel_size
        self.k = 2 * np.pi / wavelength  # wave number

    def fresnel_propagation(self, field):
        """
        Perform Fresnel propagation of a complex field.

        :param field: 2D complex array representing the field
        :return: Propagated field
        """
        ny, nx = field.shape
        fx = np.fft.fftfreq(nx, self.pixel_size)
        fy = np.fft.fftfreq(ny, self.pixel_size)
        FX, FY = np.meshgrid(fx, fy)

        # Quadratic phase factor
        H = np.exp(-1j * (self.k / (2 * self.distance)) * (FX ** 2 + FY ** 2))

        # Fourier transform, apply filter, inverse Fourier transform
        field_ft = fft2(field)
        propagated_field_ft = H * field_ft
        propagated_field = ifft2(propagated_field_ft)

        return propagated_field

    def compute_phase(self, field):
        """
        Compute the phase of the complex field.

        :param field: 2D complex array representing the field
        :return: Phase of the field
        """
        phase = np.angle(field)
        return phase

    def compute_intensity(self, field):
        """
        Compute the intensity of the complex field.

        :param field: 2D complex array representing the field
        :return: Intensity of the field
        """
        intensity = np.abs(field) ** 2
        return intensity

    def propagate_and_measure(self, field):
        """
        Perform Fresnel propagation and return phase and intensity.

        :param field: 2D complex array representing the field
        :return: Phase and intensity of the propagated field
        """
        propagated_field = self.fresnel_propagation(field)
        phase = self.compute_phase(propagated_field)
        intensity = self.compute_intensity(propagated_field)
        return phase, intensity
