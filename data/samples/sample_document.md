# Project Falcon: Autonomous Planetary Rover Architecture

## Executive Summary
Project Falcon is an autonomous exploration rover designed for high-radiation extraterrestrial terrain.
The primary mission objective is subterranean cave exploration and soil sample spectral analysis.

## Subsystems and Specifications
- **Power Subsystem**: Dual Multi-Mission Radioisotope Thermoelectric Generators (MMRTG) delivering 240 Watts of continuous electrical power, supplemented by 45 Ah lithium-ion energy storage cells.
- **Communications**: High-Gain X-band antenna capable of 2.5 Mbps transmission to orbital relay satellites.
- **Compute Architecture**: Radiation-hardened RAD750 PowerPC running at 133 MHz paired with a dedicated neural accelerator for terrain hazard classification.

## Navigation & Hazard Avoidance
The Autonomous Hazard Avoidance (AutoNav) subsystem operates at a processing frequency of 4 Hz.
It utilizes stereoscopic navigation cameras (Navcams) to compute dense disparity maps and project point clouds into a 2.5D elevation grid.

| Subsystem | Mass (kg) | Operating Temperature (°C) | Redundancy Level |
| :--- | :--- | :--- | :--- |
| Power (MMRTG) | 43.5 | -40 to +110 | Triple Modular |
| Drive Actuators | 62.0 | -120 to +50 | Dual Motor |
| Avionics Core | 18.2 | -20 to +60 | Hot Spare Dual |
| Science Suite | 31.8 | -10 to +40 | Single Sensor |

## Environmental Constraints
Maximum operational terrain slope is 32 degrees in consolidated regolith and 18 degrees in loose drift sand.
If wheel slip exceeds 40%, the autonomous drive controller engages emergency slip-mitigation maneuvers.
