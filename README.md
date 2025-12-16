To get the data run:
```
wget https://github.com/OncologyModelingGroup/TumorTwin/raw/refs/heads/main/input_files/HGG_demo_001.tar.gz
tar -xzvf HGG_demo_001.tar.gz
```


# BASIC PROMPT

I am a medical researcher working on High-Grade Glioma (HGG) treatment prediction. I need a mathematical model specification to simulate tumor evolution for a specific patient.

Patient Context:
    Diagnosis: High-Grade Glioma (HGG).
    Treatment Plan: Concurrent Radiotherapy (standard fractionation) and Chemotherapy (Temozolomide).
    Available Data: I have a pre-processed 3D cellularity map derived from MRI (normalized cell density 0≤c≤1) which I want to use as the exact initial condition at t=0.
Task:
    Propose the most appropriate Partial Differential Equation (PDE) framework to model the spatiotemporal growth of this tumor, explicitly accounting for invasion and proliferation.
    Define the mathematical terms for Radiotherapy and Chemotherapy within this PDE.
    Provide the specific equations and a Python class structure that solves this PDE on a 3D grid, using my MRI-derived cellularity map as the initial state (c0​) and accepting a daily treatment schedule as input.


# ADVANCED PROMPT

###**The Benchmark Prompt (Refined for Pharmacokinetics)**
 "I need a Python class implementing a mathematical model for **High-Grade Glioma (HGG)** prediction. The model must integrate **3D Reaction-Diffusion** tumor growth with a specific **Pharmacokinetic (PK)** module for chemotherapy.
 **Patient Data & Constraints:**
 * **Input:** A 3D numpy array representing cell density (0 \le c \le 1) derived from MRI.
 * **Domain:** 3D Cartesian grid with no-flux boundary conditions.
 
 
 **Mathematical Requirements:**
 1. **Tumor Growth:** Modeled via the Partial Differential Equation:
 
 
 
 where D is diffusivity and \rho is proliferation.
 2. **Chemotherapy (Pharmacokinetics):** Modeled not as a constant rate, but as a dynamic variable C_{\text{drug}} governed by the ODE:
 
 
 * **Key Constraint:** The class must accept a specific `decay_rate` (float) parameter. For Temozolomide, I will use \lambda \approx 9.2420 \text{ day}^{-1}, so the code must handle rapid drug clearance correctly in the time loop.
 
 
 3. **Radiotherapy:** Modeled as a discrete state update using the **Linear-Quadratic Model** (S = e^{-\alpha d - \beta d^2}) applied instantaneously at scheduled treatment days.
 
 
 **Coding Task:**
 Write a Python class `GliomaPKModel` that:
 * Initializes with parameters: D, \rho, \alpha, \beta, \text{decay\_rate}.
 * Accepts a `chemo_schedule` (list of doses/times) and an `rt_schedule` (dictionary of days/doses).
 * Solves the system using the **Finite Difference Method** for spatial diffusion and **Explicit Euler** for time integration.
 * Returns history arrays for: Time, Total Tumor Count, and **Current Drug Concentration** (so I can plot the PK spikes)."


Full conversation with gemini:
https://gemini.google.com/share/593626997543