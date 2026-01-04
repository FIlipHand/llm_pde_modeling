import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from extract_treatment_plans import extract_treatment_plans
from pydantic import FilePath
from tumortwin.preprocessing import ADC_to_cellularity
from tumortwin.types.hgg_data import HGGPatientData
from tumortwin.types import (
    CropSettings,
    CropTarget,
)
from tumortwin.postprocessing import (
    compute_total_cell_count,
    plot_cellularity_map,
    plot_predicted_TCC,
    plot_measured_TCC,
)
from tumortwin.utils import daterange, days_since_first
import torch


class GliomaModel3D:
    def __init__(self, D, rho, K, alpha, beta, decay_rate=9.2420, voxel_size_mm=1.0):
        """
        3D Reaction-Diffusion-Treatment Model.
        
        Parameters:
        D: Diffusion coefficient (mm^2/day)
        rho: Proliferation rate (1/day)
        K: Carrying capacity (normalized 0-1)
        alpha, beta: LQ Model parameters (1/Gy, 1/Gy^2)
        decay_rate: Drug elimination rate [1/day]. Matches the
            pharmacokinetic parameter used in `test.py`
            (9.2420 ≈ TMZ T_half ~ 1.8 hours).
        voxel_size_mm: The physical size of one voxel (assumed isotropic dx=dy=dz)
        """
        self.D = D
        self.rho = rho
        self.K = K
        self.alpha = alpha
        self.beta = beta
        # Keep decay_rate available on the main model so it can be
        # used for pharmacokinetic extensions consistent with `test.py`.
        self.decay_rate = decay_rate
        self.dx = voxel_size_mm
        
        # Placeholders
        self.c = None
        self.shape = None
        self.affine = np.eye(4) # For saving NIfTI later
        
        # History
        self.time_history = []
        self.tumor_count_history = []
        self.c_history = []  # Store self.c values for each day
        # Track systemic drug concentration over time (for PK model)
        self.drug_conc_history = []

    def load_initial_condition(self, nifti_data, affine=None):
        """
        Load the preprocessed 3D numpy array (cellularity map).
        nifti_data: 3D numpy array (normalized 0-1)
        affine: (Optional) 4x4 matrix from the original NIfTI file to save correct output.
        """
        self.c = nifti_data.astype(np.float32)
        self.shape = self.c.shape
        if affine is not None:
            self.affine = affine
        print(f"Model initialized with 3D grid: {self.shape}")

    def run_simulation(self, total_days, rt_schedule, chemo_schedule, dt=0.5):
        """
        Runs the 3D simulation.
        
        Chemo is modeled with a simple pharmacokinetic compartment,
        using `decay_rate` exactly as in `test.py`:
          - `chemo_schedule` is a list of (start_day, end_day, dose_amount)
          - `dose_amount` is the concentration added once per day
          - systemic concentration then decays exponentially with rate `decay_rate`
          - tumor kill term is proportional to the instantaneous concentration.
        
        Warning: 3D Explicit Euler is computationally heavy.
        """
        # Stability check (3D requires stricter dt limit: dt <= dx^2 / (6*D))
        stability_limit = (self.dx**2) / (6 * self.D)
        if dt > stability_limit:
            print(f"Warning: dt={dt} is unstable for 3D. Auto-reducing to {stability_limit*0.9:.4f}")
            dt = stability_limit * 0.9

        steps = int(total_days / dt)
        print(f"Simulating {total_days} days in 3D ({steps} steps)...")
        
        # Initialize systemic drug concentration for PK model
        current_drug_conc = 0.0
        
        # Store initial condition (day 0)
        self.c_history.append(self.c.copy())
        
        for step in range(steps):
            t = step * dt

            # --- 1. Pharmacokinetics (Drug Decay) ---
            # Exponential decay: C(t+dt) = C(t) * exp(-lambda * dt)
            current_drug_conc *= np.exp(-self.decay_rate * dt)

            # --- 2. Day boundary detection (for dosing & RT) ---
            current_day_int = int(t)
            prev_day_int = int(t - dt)
            is_new_day = current_day_int > prev_day_int

            # --- 3. Chemotherapy Dosing (Daily Spikes) ---
            if is_new_day:
                for (start, end, dose) in chemo_schedule:
                    if start <= t <= end:
                        current_drug_conc += dose

            # --- 4. 3D Diffusion (7-point Stencil Laplacian) ---
            # We use np.roll to shift the array in all 6 directions (x, y, z)
            # x-axis
            c_left  = np.roll(self.c, -1, axis=0)
            c_right = np.roll(self.c, 1, axis=0)
            # y-axis
            c_back  = np.roll(self.c, -1, axis=1)
            c_front = np.roll(self.c, 1, axis=1)
            # z-axis
            c_down  = np.roll(self.c, -1, axis=2)
            c_up    = np.roll(self.c, 1, axis=2)

            # 3D Laplacian: (Sum of neighbors - 6*center) / dx^2
            laplacian = (c_left + c_right + c_back + c_front + c_down + c_up - 6*self.c) / (self.dx**2)

            # --- 5. Reaction Terms ---
            diffusion_term = self.D * laplacian
            proliferation_term = self.rho * self.c * (1 - self.c / self.K)

            # --- 6. Chemo Effect (Concentration Dependent) ---
            # Kill is now proportional to current drug concentration: -C(t) * c
            chemo_kill = current_drug_conc * self.c

            # Apply update
            self.c += dt * (diffusion_term + proliferation_term - chemo_kill)

            # --- 7. Radiotherapy (Discrete) ---
            if is_new_day and (current_day_int in rt_schedule):
                dose = rt_schedule[current_day_int]
                # Note: We assume uniform dose here. 
                # If you have a 3D DOSE MAP, replace 'dose' with the dose_array.
                sf = np.exp(-(self.alpha * dose + self.beta * dose**2))
                self.c *= sf
                # print(f"Day {current_day_int}: RT Applied")

            # --- 8. Boundary Conditions (No Flux) ---
            # Simply reset the outer edges to their inner neighbors to prevent wrap-around artifacts from np.roll
            self.c[0,:,:] = self.c[1,:,:]; self.c[-1,:,:] = self.c[-2,:,:]
            self.c[:,0,:] = self.c[:,1,:]; self.c[:,-1,:] = self.c[:,-2,:]
            self.c[:,:,0] = self.c[:,:,1]; self.c[:,:,-1] = self.c[:,:,-2]

            # Clamp limits
            self.c = np.clip(self.c, 0, self.K)

            # Record daily snapshot for c_history when a new day starts
            if is_new_day:
                self.c_history.append(self.c.copy())

            # Record metrics
            self.time_history.append(t)
            self.tumor_count_history.append(compute_total_cell_count(self.c, self.K))
            self.drug_conc_history.append(current_drug_conc)
            
            # Optional: Print progress every 10 days
            if step % (10/dt) < 1: 
                print(f"  Day {t:.1f}: Total Burden = {compute_total_cell_count(self.c, self.K):.1f}")

        # Store the final day's c value if not already stored
        final_day = int(total_days)
        if len(self.c_history) == 0 or int(self.time_history[-1]) < final_day:
            self.c_history.append(self.c.copy())

        # Keep return signature compatible with existing code
        return self.time_history, self.c_history
# ==========================================
#      USER CONFIGURATION SECTION
# ==========================================

# 1. Define Patient Parameters
D_patient = 0.025   # Diffusivity (mm^2/day)
rho_patient = 0.05  # Proliferation (1/day)
K_capacity = 1.0    # Max density

# 2. Define Treatment Schedules
# RT: Dictionary { Day : Dose_Gy }
# Example: 2Gy daily from day 10 to 15
# rt_plan = {
#     10: 2.0, 11: 2.0, 12: 2.0, 13: 2.0, 14: 2.0,
#     17: 2.0, 18: 2.0, 19: 2.0, 20: 2.0, 21: 2.0  # Second week
# }

# Chemo: List of [(Start_Day, End_Day, Kill_Rate)]
# Example: Drug active from day 10 to day 25 with 0.01 kill rate
# chemo_plan = [
    # (10, 25, 0.01) 
# ]
PATIENT_INFO_PATH = FilePath("./HGG_demo_001/HGG_demo_001.json")
IMAGE_PATH = FilePath("./HGG_demo_001")
crop_settings = CropSettings(crop_to=CropTarget.ROI_ENHANCE, padding=10, visit_index=-1)
patient_data = HGGPatientData.from_file(
    PATIENT_INFO_PATH, image_dir=IMAGE_PATH, crop_settings=crop_settings
)

measured_cellularity_maps = [
    ADC_to_cellularity(
        visit.adc_image, visit.roi_enhance_image, visit.roi_nonenhance_image
    )
    for visit in patient_data.visits
]

rt_plan, chemo_plan = extract_treatment_plans("./HGG_demo_001/HGG_demo_001.json")

model = GliomaModel3D(D=0.025, rho=0.05, K=1.0, alpha=0.035, beta=0.0035, voxel_size_mm=1.0)

# 4. LOAD DATA
model.load_initial_condition(measured_cellularity_maps[0].array, affine=np.eye(4))

# 5. RUN
# Note: 3D takes longer. Start with small total_days to test.
time, c_history = model.run_simulation(total_days=270, rt_schedule=rt_plan, chemo_schedule=chemo_plan)
print(f"Number of days stored: {len(c_history)}")


# Compute tumor counts from c arrays for plotting
tumor_counts = [compute_total_cell_count(c, 5062500) for c in c_history] # value from docs
history_file = "./experiment_data/HGG_llm.npy"
np.save(history_file, tumor_counts)
exit()
# Create day numbers corresponding to c_history
days = list[int](range(len(c_history)))
print(days)
print(tumor_counts)

# # 6. Plot Results
plt.figure(figsize=(10, 5))
plt.plot(days, tumor_counts, linewidth=2, color='darkblue', label='Predicted Tumor Cell Count')
plt.title(f"Treatment Response Prediction\n(D={D_patient}, rho={rho_patient})")
plt.xlabel("Days")
plt.ylabel("Tumor cell count")
plt.grid(True, linestyle='--', alpha=0.7)

# Highlight all radiotherapy sessions
if rt_plan:
    rt_days = sorted(rt_plan.keys())
    # Draw vertical lines for each RT session
    for rt_day in rt_days:
        plt.axvline(x=rt_day, color='red', alpha=0.3, linewidth=1)
    # Draw a span covering all RT days for visual clarity
    if len(rt_days) > 0:
        plt.axvspan(min(rt_days), max(rt_days), color='red', alpha=0.1, 
                   label=f'Radiotherapy ({len(rt_days)} sessions)')

# Highlight all chemotherapy cycles
if chemo_plan:
    # Show all chemo cycles with spans
    for i, (c_start, c_end, kill_rate) in enumerate(chemo_plan):
        # Only label the first cycle to avoid legend clutter
        label = f'Chemotherapy ({len(chemo_plan)} cycles)' if i == 0 else ''
        plt.axvspan(c_start, c_end, color='green', alpha=0.15, label=label)

plt.legend()
plt.tight_layout()
plt.show()

# Create timepoints matching c_history (one per day)
# c_history has entries for days 0, 1, 2, ..., so create timepoints accordingly
first_visit_time = patient_data.visits[0].time
timepoints = [first_visit_time + timedelta(days=i) for i in range(len(c_history))]

# Convert c_history to torch tensors for plotting
predicted_cellularity_maps = [torch.tensor(c) for c in c_history]

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