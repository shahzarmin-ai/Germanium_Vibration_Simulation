#Germanium_Vibration_Simulation
"""
Simulation of Germanium Segregation during Zone Refining with Vibration Monitoring
Author: Zarmina Shah

- Simulates Germanium concentration profile along a rod using Scheil-Pfann equation
- Monte Carlo uncertainty analysis
- Simulates mechanical vibrations induced during zone refining
- Virtual accelerometers measure vibrations along the rod
- FFT-based analysis for anomaly detection

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# ----------------------------
# 1. Germanium Segregation 
# ----------------------------
def scheil_pfann(C0, k, L):
    """
    Calculate concentration along a rod using Scheil-Pfann equation.
    C0: initial concentration
    k: segregation coefficient
    L: normalized length along the rod (0 to 1)
    """
    return C0 * (1 - L)**(k - 1)

# Parameters
C0 = 0.02       # initial Ge fraction
k = 0.3         # segregation coefficient
N_points = 500  # points along the rod
L = np.linspace(0, 1, N_points)
C_profile = scheil_pfann(C0, k, L)

# Monte Carlo variation
MC_runs = 5
C_profiles_MC = []
for _ in range(MC_runs):
    noise = np.random.normal(0, 0.001, N_points)
    C_profiles_MC.append(C_profile + noise)
C_profiles_MC = np.array(C_profiles_MC)

# ----------------------------
# 2. Vibration Simulation 
# ----------------------------
m = 2.0       # mass of rod segment (kg)
k_vib = 1000  # stiffness (N/m)
c = 2.0       # damping coefficient
dt = 0.001
T = 2         # total time (s)
N_time = int(T/dt)
t = np.linspace(0, T, N_time)

# Force induced by moving heater during zone refining
F = 5 * np.sin(2 * np.pi * 3 * t)  # periodic forcing

x = np.zeros(N_time)
v = np.zeros(N_time)
a = np.zeros(N_time)

for i in range(1, N_time):
    a[i] = (F[i] - c*v[i-1] - k_vib*x[i-1]) / m
    v[i] = v[i-1] + a[i]*dt
    x[i] = x[i-1] + v[i]*dt

# Virtual accelerometer with noise
acc_noise_std = 0.05
acc_sensor = a + np.random.normal(0, acc_noise_std, N_time)

# ----------------------------
# 3. FFT Analysis
# ----------------------------
yf = fft(acc_sensor)
xf = fftfreq(N_time, dt)[:N_time//2]

# ----------------------------
# 4. Visualization
# ----------------------------
# Germanium segregation profiles
plt.figure(figsize=(12,5))
plt.plot(L, C_profile, label="Nominal")
for i in range(MC_runs):
    plt.plot(L, C_profiles_MC[i], linestyle='--', alpha=0.5)
plt.xlabel("Normalized Rod Length")
plt.ylabel("Ge Fraction")
plt.title("Germanium Segregation Profile (Scheil-Pfann)")
plt.legend()
plt.grid(True)
plt.show()

# Vibration time series
plt.figure(figsize=(12,4))
plt.plot(t, acc_sensor)
plt.title("Virtual Accelerometer Signal During Zone Refining")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.grid(True)
plt.show()

# FFT spectrum
plt.figure(figsize=(12,4))
plt.plot(xf, 2.0/N_time * np.abs(yf[0:N_time//2]))
plt.title("Frequency Spectrum of Vibration Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

print("Simulation completed: Germanium segregation and vibration monitoring executed.")
