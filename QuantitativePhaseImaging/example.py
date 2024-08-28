from qpi import QuantitativePhaseImaging
import numpy as np
import matplotlib.pyplot as plt

qpi = QuantitativePhaseImaging(wavelength=500e-9, distance=0.1, pixel_size=1e-6)

ny, nx = 512, 512
x = np.linspace(-1, 1, nx) * nx * qpi.pixel_size
y = np.linspace(-1, 1, ny) * ny * qpi.pixel_size
X, Y = np.meshgrid(x, y)
field = np.exp(- (X**2 + Y**2) / (2 * (0.1)**2)) * np.exp(1j * qpi.k * (X**2 + Y**2) / (2 * 0.1))

phase, intensity = qpi.propagate_and_measure(field)

plt.figure()
plt.imshow(phase, cmap='gray')
plt.title('Phase')

plt.figure()
plt.imshow(intensity, cmap='gray')
plt.title('Intensity')

plt.show()
