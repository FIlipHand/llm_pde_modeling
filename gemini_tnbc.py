import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from typing import Tuple
from extract_treatment_plans import extract_treatment_plans
from pydantic import FilePath
from tumortwin.preprocessing import ADC_to_cellularity
from tumortwin.types.tnbc_data import TNBCPatientData
from tumortwin.types import (
    ChemotherapySpecification,
    CropSettings,
    CropTarget,
)
from tumortwin.models import ReactionDiffusion3D
from tumortwin.postprocessing import (
    compute_total_cell_count,
    plot_cellularity_map,
    plot_predicted_TCC,
    plot_measured_TCC,
)
from tumortwin.utils import daterange, days_since_first
import torch
from pprint import pprint

class TNBCSimulator:
    def __init__(self, c0, D, rho, K, sensitivity, decay_rate, voxel_size_mm=1.0):
        self.c = np.copy(c0)      # 3D Cellularity Map
        self.c = np.clip(c0, 0, K) 
        self.K = K  # Carrying capacity (e.g., 0.8 for 80% max tumor density)
        # Parameters
        self.D = D                # Diffusion (mm^2/day)
        self.rho = rho            # Proliferation (1/day)
        self.s = sensitivity      # s = 0.2
        self.d = decay_rate       # d = 0.7
        self.dx = voxel_size_mm   # Voxel size for Laplacian calculation
        self.k_tumor = 0.0        # Drug concentration (initialized to 0)

    def _laplacian_3d(self, f):
        """Standard 7-point stencil for 3D Laplacian."""
        lap = -6 * f.copy()
        lap += np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0)
        lap += np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1)
        lap += np.roll(f, 1, axis=2) + np.roll(f, -1, axis=2)
        return lap / (self.dx**2)

    def step(self, dt, dose_amount):
        # 1. Update Drug Concentration
        # Add dose and apply exponential decay
        self.k_tumor += dose_amount
        self.k_tumor *= np.exp(-self.d * dt)
        
        # 2. Update Tumor Density with Carrying Capacity K
        diffusion = self.D * self._laplacian_3d(self.c)
        
        # Modified Proliferation: Logistic growth relative to K
        proliferation = self.rho * self.c * (1 - (self.c / self.K))
        
        death = self.s * self.k_tumor * self.c
        
        # Update State
        self.c += dt * (diffusion + proliferation - death)
        
        # 3. Hard Physical Boundary
        self.c = np.clip(self.c, 0, self.K)

    def simulate(self, total_days, dt, chemo_schedule):
        """
        chemo_schedule: list of tuples (start_day, end_day, kill_rate)
        """
        steps_per_day = int(1 / dt)
        history = []
        
        for day in range(total_days):
            for step in range(steps_per_day):
                # Check if this day is within any chemo treatment period
                dose_amount = 0.0
                if step == 0:  # Only check at the start of each day
                    for (start, end, kill_rate) in chemo_schedule:
                        if start <= day <= end:
                            dose_amount = kill_rate * 1000
                            break
                
                self.step(dt, dose_amount)
            
            history.append(np.copy(self.c))
            
        return np.array(history)


if __name__ == "__main__":
    PATIENT_INFO_PATH = FilePath("./TNBC_demo_001/TNBC_demo_001.json")
    IMAGE_PATH = FilePath("./TNBC_demo_001")
    crop_settings = CropSettings(crop_to=CropTarget.ROI_ENHANCE, padding=10, visit_index=-1)
    patient_data = TNBCPatientData.from_file(
        PATIENT_INFO_PATH, image_dir=IMAGE_PATH, crop_settings=crop_settings
    )

    # ct = ChemotherapySpecification(
    #     sensitivity=0.2,
    #     decay_rate=0.7,
    #     times=[c.time for c in patient_data.chemotherapy],
    #     doses=[c.dose for c in patient_data.chemotherapy],
    # )
    # print(ct)
    # exit(1)

    measured_cellularity_maps = [
    ADC_to_cellularity(
        visit.adc_image, visit.roi_enhance_image
    )
    for visit in patient_data.visits
    ]

    _, chemo_plan = extract_treatment_plans("TNBC_demo_001/TNBC_demo_001.json")
    
    D_patient = 0.025   # Diffusivity (mm^2/day)
    rho_patient = 0.05  # Proliferation (1/day)
    K_capacity = 1.0    # Max density
    sensitivity = 0.2
    decay_rate = 0.7
    model = TNBCSimulator(c0=measured_cellularity_maps[0].array, D=D_patient, rho=rho_patient, K=K_capacity, sensitivity=sensitivity, decay_rate=decay_rate)
    
    history = model.simulate(120, 1, chemo_plan)

    # tumor_counts = [compute_total_cell_count(c, model.K) for c in history] # value from docs
    # history_file = "./experiment_data/TNBC_llm.npy"
    # print(len(tumor_counts))
    # np.save(history_file, tumor_counts)
    # exit()
    # Load or run simulation
    
    tumor_counts = [compute_total_cell_count(c, model.K) for c in history]
    days = list[int](range(len(history)))


    # plt.figure(figsize=(10, 5))
    # plt.plot(days, tumor_counts, linewidth=2, color='darkblue', label='Predicted Tumor Cell Count')
    # plt.title(f"Treatment Response Prediction\n(D={D_patient}, rho={rho_patient})")
    # plt.xlabel("Days")
    # plt.ylabel("Tumor cell count")
    # plt.grid(True, linestyle='--', alpha=0.7)

    # # Highlight all chemotherapy sessions
    # if chemo_plan:
    #     # Show all chemo cycles with spans
    #     for i, (c_start, c_end, kill_rate) in enumerate[Tuple[int, int, float]](chemo_plan):
    #         # Only label the first cycle to avoid legend clutter
    #         label = f'Chemotherapy ({len(chemo_plan)} cycles)' if i == 0 else ''
    #         plt.axvspan(c_start, c_end, color='green', alpha=0.15, label=label)

    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # Create timepoints matching c_history (one per day)
    # c_history has entries for days 0, 1, 2, ..., so create timepoints accordingly
    first_visit_time = patient_data.visits[0].time
    timepoints = [first_visit_time + timedelta(days=i) for i in range(len(history))]

    # Convert c_history to torch tensors for plotting
    predicted_cellularity_maps = [torch.tensor(c) for c in history]
    predicted_cell_counts = [
        compute_total_cell_count(N, 5062500)
        for N in predicted_cellularity_maps
    ]
    history_file = "./experiment_data/TNBC_llm.npy"
    np.save(history_file, predicted_cell_counts)

    # plot TCC vs. time for predictions and measurements
    fig, ax = plt.subplots(1, 1, figsize=(5, 2))
    plot_predicted_TCC(predicted_cellularity_maps, timepoints, ax=ax)
    plot_measured_TCC(
        [m.array for m in measured_cellularity_maps],
        [v.time for v in patient_data.visits],
        ax=ax,
    )
    ax.legend(["predicted", "measured"])

    # plot cellularity maps for predictions and measurements
    fig, axs = plt.subplots(2, len(patient_data.visit_days[::2]), figsize=(5, 2))
    for i, t in enumerate[float](patient_data.visit_days[::2]):
        time_days = np.array([days_since_first(tp, timepoints[0]) for tp in timepoints])
        t_idx = np.where(time_days == t)[0][0]
        plot_cellularity_map(
            predicted_cellularity_maps[t_idx], patient_data, time=t, ax=axs[0, i]
        )
        plot_cellularity_map(
            torch.tensor(measured_cellularity_maps[2 * i].array), patient_data, time=t, ax=axs[1, i]
        )
    axs[0, 0].set_ylabel("Predicted")
    axs[1, 0].set_ylabel("Measured")
    plt.show()