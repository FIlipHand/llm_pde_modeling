import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
from pydantic import FilePath

from tumortwin.preprocessing import ADC_to_cellularity
from tumortwin.types.hgg_data import HGGPatientData
from tumortwin.types.tnbc_data import TNBCPatientData
from tumortwin.types import CropSettings, CropTarget
from tumortwin.postprocessing import compute_total_cell_count
from tumortwin.utils import days_since_first


def load_patient_data(mode: Literal['hgg', 'tnbc']):
    """Load patient data and compute measured cellularity maps."""
    crop_settings = CropSettings(crop_to=CropTarget.ROI_ENHANCE, padding=10, visit_index=-1)
    
    match mode:
        case 'hgg':
            patient_info_path = FilePath("./HGG_demo_001/HGG_demo_001.json")
            image_path = FilePath("./HGG_demo_001")
            patient_data = HGGPatientData.from_file(
                patient_info_path, image_dir=image_path, crop_settings=crop_settings
            )
            measured_cellularity_maps = [
                ADC_to_cellularity(
                    visit.adc_image, visit.roi_enhance_image, visit.roi_nonenhance_image
                )
                for visit in patient_data.visits
            ]
        case 'tnbc':
            patient_info_path = FilePath("./TNBC_demo_001/TNBC_demo_001.json")
            image_path = FilePath("./TNBC_demo_001")
            patient_data = TNBCPatientData.from_file(
                patient_info_path, image_dir=image_path, crop_settings=crop_settings
            )
            measured_cellularity_maps = [
                ADC_to_cellularity(
                    visit.adc_image, visit.roi_enhance_image
                )
                for visit in patient_data.visits
            ]
        case _:
            raise ValueError(f"Unknown mode: {mode}")
    
    return patient_data, measured_cellularity_maps


def cell_count_comparison(mode: Literal['hgg', 'tnbc'], path: str | None = None):
    """
    Compare cell counts between original model and LLM-generated model.
    
    Args:
        mode: Either 'hgg' or 'tnbc' to select the patient type
        path: Optional path to save the plot. If None, the plot is displayed.
    """
    # Load data file paths
    match mode:
        case 'hgg':
            filename_orig = './experiment_data/HGG_original.npy'
            filename_llm = './experiment_data/HGG_llm.npy'
        case 'tnbc':
            filename_orig = './experiment_data/TNBC_original.npy'
            filename_llm = './experiment_data/TNBC_llm.npy'
        case _:
            raise ValueError(f"Unknown mode: {mode}")
    
    # Load patient data
    patient_data, measured_cellularity_maps = load_patient_data(mode)
    
    # Load model predictions
    orig = np.load(filename_orig)
    llm = np.load(filename_llm)
    
    # Create days array for x-axis
    days = np.arange(len(orig))
    
    # Compute measured total cell counts
    carrying_capacity = 5062500  # From calibration_summary.py default
    measured_cell_counts = [
        compute_total_cell_count(m.array, carrying_capacity)
        for m in measured_cellularity_maps
    ]
    
    # Get timepoints for measured data
    visit_timepoints = [visit.time for visit in patient_data.visits]
    first_visit_time = visit_timepoints[0]
    measured_days = [
        days_since_first(t, first_visit_time) for t in visit_timepoints
    ]
    
    # Extract treatment schedules
    rt_days = []
    if hasattr(patient_data, 'radiotherapy') and patient_data.radiotherapy:
        rt_days = [
            days_since_first(rt.time, first_visit_time)
            for rt in patient_data.radiotherapy
        ]
    
    chemo_ranges = []
    if hasattr(patient_data, 'chemotherapy') and patient_data.chemotherapy:
        # Group consecutive chemotherapy days into ranges
        chemo_entries = [
            (days_since_first(c.time, first_visit_time), c.dose)
            for c in patient_data.chemotherapy
        ]
        chemo_entries.sort(key=lambda x: x[0])
        
        if chemo_entries:
            current_start = chemo_entries[0][0]
            current_dose = chemo_entries[0][1]
            
            for i in range(1, len(chemo_entries)):
                day, dose = chemo_entries[i]
                # If dose changed or gap > 1 day, start new range
                if dose != current_dose or day != chemo_entries[i-1][0] + 1:
                    end_day = chemo_entries[i-1][0]
                    chemo_ranges.append((current_start, end_day))
                    current_start = day
                    current_dose = dose
            
            # Add final range
            end_day = chemo_entries[-1][0]
            chemo_ranges.append((current_start, end_day))
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot treatment backgrounds first (so they appear behind the data)
    if rt_days:
        rt_days_sorted = sorted(rt_days)
        # Draw a span with diagonal stripes for radiotherapy
        if len(rt_days_sorted) > 0:
            plt.axvspan(
                min(rt_days_sorted), max(rt_days_sorted),
                color='red', alpha=0.15,
                hatch='///',  # Diagonal stripes from left to right
                edgecolor='red',
                linewidth=0.5,
                label=f'Radiotherapy ({len(rt_days_sorted)} sessions)'
            )
    
    if chemo_ranges:
        for i, (c_start, c_end) in enumerate(chemo_ranges):
            # Only label the first cycle to avoid legend clutter
            label = f'Chemotherapy ({len(chemo_ranges)} cycles)' if i == 0 else ''
            plt.axvspan(c_start, c_end, color='green', alpha=0.15, label=label)
    
    # Plot data on top
    plt.plot(days, orig, label="Model from package", linewidth=2) 
    plt.plot(days, llm, label="Model created with LLM", linewidth=2)
    plt.plot(
        measured_days,
        measured_cell_counts,
        marker="X",
        markersize=6,
        color="r",
        linestyle="None",
        label="Measured",
    )
    plt.xlabel('Days of Treatment', fontsize=12)
    plt.ylabel('Total Cell Count', fontsize=12)
    plt.title(f'Cell Count Comparison - {mode.upper()}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    cell_count_comparison('tnbc', './tnbc_comparison.png')
    cell_count_comparison('hgg', './hgg_comparison.png')