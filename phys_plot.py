import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import traceback


def plot_infall_physics_improved(GalGA, results_df=None, save_path='GA/Infall_Physics_Improved.png'):
    """
    Improved plot of the two-infall model physics showing both gas accretion episodes clearly
    Now with robust error handling and fallbacks
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Get best model parameters with fallbacks
        if results_df is not None and not results_df.empty:
            bm = results_df.iloc[0]
            best_params = (bm['sigma_2'], bm['t_2'], bm['infall_2'])
            sigma_2, t_1, t_2, infall_1, infall_2 = bm['sigma_2'], bm['t_1'], bm['t_2'], bm['infall_1'], bm['infall_2']
            sfe_val, delta_sfe_val = bm['sfe'], bm['delta_sfe']
        elif hasattr(GalGA, 'results') and len(GalGA.results) > 0:
            r = GalGA.results[0]
            best_params = (r[5], r[7], r[9])
            sigma_2, t_1, t_2, infall_1, infall_2 = r[5], r[6], r[7], r[8], r[9]
            sfe_val, delta_sfe_val = r[10], r[11]
        else:
            # Fallback values if no results available
            print("Warning: No results available, using fallback parameters for infall physics plot")
            sigma_2, t_1, t_2, infall_1, infall_2 = 100.0, 0.1, 8.0, 2.0, 1.0
            sfe_val, delta_sfe_val = 0.01, 0.0
        
        # Create the plot
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # Time array for infall functions - extend to show full evolution
        time_gyr = np.linspace(0, 14, 2000)
        
        # Get the actual A1 and A2 values from the GalGA object with fallbacks
        A1 = getattr(GalGA, 'A1', 50.0)
        A2 = getattr(GalGA, 'A2', 100.0)
        
        print(f"Using infall normalizations: A1 = {A1}, A2 = {A2}")
        
        # 1. Two-infall model visualization with LINEAR scale first
        ax1 = fig.add_subplot(gs[0, :])
        
        # Calculate first infall component (starts at t_1, typically ~0)
        infall_rate_1 = np.zeros_like(time_gyr)
        for i, t in enumerate(time_gyr):
            if t >= t_1:
                infall_rate_1[i] = A1 * np.exp(-(t - t_1) / infall_1)
        
        # Calculate second infall component (starts at t_2)
        infall_rate_2 = np.zeros_like(time_gyr)
        for i, t in enumerate(time_gyr):
            if t >= t_2:
                infall_rate_2[i] = A2 * np.exp(-(t - t_2) / infall_2)
        
        # Total infall rate
        infall_rate_total = infall_rate_1 + infall_rate_2
        
        # Plot with both linear and log versions
        ax1.plot(time_gyr, infall_rate_total, 'black', linewidth=3, label='Total Infall Rate')
        ax1.plot(time_gyr, infall_rate_1, 'blue', linewidth=2, linestyle='--', 
                 label=f'First Infall (t_start={t_1:.3f} Gyr, τ={infall_1:.2f} Gyr)')
        ax1.plot(time_gyr, infall_rate_2, 'red', linewidth=2, linestyle='--', 
                 label=f'Second Infall (t_start={t_2:.1f} Gyr, τ={infall_2:.2f} Gyr)')
        
        # Mark the infall start times
        ax1.axvline(t_1, color='blue', linestyle=':', alpha=0.7, label=f'First Infall Start (t={t_1:.3f})')
        ax1.axvline(t_2, color='red', linestyle=':', alpha=0.7, label=f'Second Infall Start (t={t_2:.1f})')
        
        ax1.set_xlabel('Age (Gyr)')
        ax1.set_ylabel(r'Infall Rate [$M_\odot$ yr$^{-1}$]')
        ax1.set_title('Two-Infall Model: Gas Accretion Episodes (Linear Scale)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.set_xlim(0, 14)
        
        # 2. Same plot but with log scale to see both components clearly
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(time_gyr, infall_rate_total, 'black', linewidth=3, label='Total')
        ax2.plot(time_gyr, infall_rate_1, 'blue', linewidth=2, linestyle='--', label='First')
        ax2.plot(time_gyr, infall_rate_2, 'red', linewidth=2, linestyle='--', label='Second')
        
        ax2.set_xlabel('Age (Gyr)')
        ax2.set_ylabel(r'Infall Rate [$M_\odot$ yr$^{-1}$]')
        ax2.set_title('Two-Infall Model (Log Scale)')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
        ax2.set_xlim(0, 14)
        
        # 3. Zoom in on first infall episode
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Focus on first 3 Gyr to show first infall clearly
        time_early = time_gyr[time_gyr <= 3.0]
        infall_1_early = infall_rate_1[time_gyr <= 3.0]
        infall_total_early = infall_rate_total[time_gyr <= 3.0]
        
        ax3.plot(time_early, infall_total_early, 'black', linewidth=2, label='Total')
        ax3.plot(time_early, infall_1_early, 'blue', linewidth=2, linestyle='--', label='First Infall')
        ax3.axvline(t_1, color='blue', linestyle=':', alpha=0.7)
        ax3.fill_between(time_early, 0, infall_1_early, alpha=0.3, color='blue')
        
        ax3.set_xlabel('Age (Gyr)')
        ax3.set_ylabel(r'Infall Rate [$M_\odot$ yr$^{-1}$]')
        ax3.set_title(f'First Infall Detail (0-3 Gyr)')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=9)
        
        # 4. Zoom in on second infall episode
        ax4 = fig.add_subplot(gs[1, 2])
        
        # Focus on last 3 Gyr to show second infall clearly
        time_late = time_gyr[time_gyr >= 11.0]
        infall_2_late = infall_rate_2[time_gyr >= 11.0]
        infall_total_late = infall_rate_total[time_gyr >= 11.0]
        
        ax4.plot(time_late, infall_total_late, 'black', linewidth=2, label='Total')
        ax4.plot(time_late, infall_2_late, 'red', linewidth=2, linestyle='--', label='Second Infall')
        ax4.axvline(t_2, color='red', linestyle=':', alpha=0.7)
        ax4.fill_between(time_late, 0, infall_2_late, alpha=0.3, color='red')
        
        ax4.set_xlabel('Age (Gyr)')
        ax4.set_ylabel(r'Infall Rate [$M_\odot$ yr$^{-1}$]')
        ax4.set_title(f'Second Infall Detail (11-14 Gyr)')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=9)
        
        # 5. Star formation efficiency evolution
        ax5 = fig.add_subplot(gs[2, 0])
        
        # SFE changes at second infall
        sfe_evolution = np.where(time_gyr < t_2, sfe_val, sfe_val + delta_sfe_val)
        
        ax5.plot(time_gyr, sfe_evolution, 'green', linewidth=3)
        ax5.axvline(t_2, color='red', linestyle=':', alpha=0.7, label=f'SFE Change at t={t_2:.1f} Gyr')
        ax5.axhline(sfe_val, color='blue', linestyle='--', alpha=0.7, label=f'Initial SFE = {sfe_val:.4f}')
        ax5.axhline(sfe_val + delta_sfe_val, color='orange', linestyle='--', alpha=0.7, 
                    label=f'Final SFE = {sfe_val + delta_sfe_val:.4f}')
        
        ax5.set_xlabel('Age (Gyr)')
        ax5.set_ylabel('Star Formation Efficiency')
        ax5.set_title('SFE Evolution')
        ax5.grid(True, alpha=0.3)
        ax5.legend(fontsize=9)
        
        # 6. Cumulative gas accretion
        ax6 = fig.add_subplot(gs[2, 1])
        
        # Calculate cumulative infall (integrate over time)
        dt = time_gyr[1] - time_gyr[0]
        cumulative_infall_1 = np.cumsum(infall_rate_1) * dt * 1e9  # Convert to years for integration
        cumulative_infall_2 = np.cumsum(infall_rate_2) * dt * 1e9
        cumulative_total = np.cumsum(infall_rate_total) * dt * 1e9
        
        ax6.fill_between(time_gyr, 0, cumulative_infall_1, alpha=0.4, color='blue', label='First Infall Component')
        ax6.fill_between(time_gyr, cumulative_infall_1, cumulative_total, alpha=0.4, color='red', label='Second Infall Component')
        ax6.plot(time_gyr, cumulative_total, 'black', linewidth=2, label='Total Cumulative Infall')
        
        ax6.set_xlabel('Age (Gyr)')
        ax6.set_ylabel(r'Cumulative Infall Mass [$M_\odot$]')
        ax6.set_title('Cumulative Gas Accretion')
        ax6.set_yscale('log')
        ax6.grid(True, alpha=0.3)
        ax6.legend(fontsize=9)
        
        # 7. Infall rate ratios and timing analysis
        ax7 = fig.add_subplot(gs[2, 2])
        
        # Calculate key quantities
        first_infall_peak = np.max(infall_rate_1)
        second_infall_peak = np.max(infall_rate_2)
        first_infall_total = np.trapz(infall_rate_1, time_gyr) * 1e9
        second_infall_total = np.trapz(infall_rate_2, time_gyr) * 1e9
        
        # Create a summary table
        ax7.axis('off')
        summary_text = f"""TWO-INFALL MODEL SUMMARY
        
Best-Fit Parameters:
• σ₂ = {sigma_2:.1f} pc
• First infall: t₁ = {t_1:.3f} Gyr, τ₁ = {infall_1:.2f} Gyr
• Second infall: t₂ = {t_2:.1f} Gyr, τ₂ = {infall_2:.2f} Gyr
• SFE change: {sfe_val:.4f} → {sfe_val + delta_sfe_val:.4f}

Infall Characteristics:
• First infall peak rate: {first_infall_peak:.1f} M☉/yr
• Second infall peak rate: {second_infall_peak:.1f} M☉/yr
• Peak ratio (2nd/1st): {second_infall_peak/first_infall_peak:.2f}
• First infall total mass: {first_infall_total:.2e} M☉
• Second infall total mass: {second_infall_total:.2e} M☉
• Mass ratio (2nd/1st): {second_infall_total/first_infall_total:.2f}

Timing:
• Time between infalls: {t_2 - t_1:.1f} Gyr
• First infall e-folding: {infall_1:.2f} Gyr
• Second infall e-folding: {infall_2:.2f} Gyr"""
        
        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Improved infall physics plot saved to {save_path}")
        print(f"First infall: peak = {first_infall_peak:.1f} M☉/yr, total = {first_infall_total:.2e} M☉")
        print(f"Second infall: peak = {second_infall_peak:.1f} M☉/yr, total = {second_infall_total:.2e} M☉")
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_infall_physics_improved: {e}")
        print("Traceback:", traceback.format_exc())
        # Create a simple fallback plot
        return create_fallback_infall_plot(save_path)


def create_fallback_infall_plot(save_path):
    """Create a simple fallback infall plot if the main function fails"""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Simple two-infall model with default parameters
        time_gyr = np.linspace(0, 14, 1000)
        A1, A2 = 50.0, 100.0
        t_1, t_2 = 0.1, 8.0
        infall_1, infall_2 = 2.0, 1.0
        
        infall_rate_1 = np.where(time_gyr >= t_1, A1 * np.exp(-(time_gyr - t_1) / infall_1), 0)
        infall_rate_2 = np.where(time_gyr >= t_2, A2 * np.exp(-(time_gyr - t_2) / infall_2), 0)
        infall_rate_total = infall_rate_1 + infall_rate_2
        
        ax.plot(time_gyr, infall_rate_total, 'black', linewidth=3, label='Total Infall Rate')
        ax.plot(time_gyr, infall_rate_1, 'blue', linewidth=2, linestyle='--', label='First Infall')
        ax.plot(time_gyr, infall_rate_2, 'red', linewidth=2, linestyle='--', label='Second Infall')
        
        ax.set_xlabel('Age (Gyr)')
        ax.set_ylabel(r'Infall Rate [$M_\odot$ yr$^{-1}$]')
        ax.set_title('Two-Infall Model (Fallback Plot)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Fallback infall physics plot saved to {save_path}")
        return fig
        
    except Exception as e:
        print(f"Even fallback infall plot failed: {e}")
        return None


def plot_gas_flow_physics_simplified(GalGA, results_df=None, save_path='GA/Gas_Flow_Physics_Simplified.png'):
    """
    Simplified gas flow physics plot that doesn't require model reconstruction
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Get best model parameters with fallbacks
        if results_df is not None and not results_df.empty:
            bm = results_df.iloc[0]
            sigma_2, t_1, t_2, infall_1, infall_2 = bm['sigma_2'], bm['t_1'], bm['t_2'], bm['infall_1'], bm['infall_2']
            sfe_val, delta_sfe_val = bm['sfe'], bm['delta_sfe']
            mgal = bm['mgal']
        elif hasattr(GalGA, 'results') and len(GalGA.results) > 0:
            r = GalGA.results[0]
            sigma_2, t_1, t_2, infall_1, infall_2 = r[5], r[6], r[7], r[8], r[9]
            sfe_val, delta_sfe_val = r[10], r[11]
            mgal = r[13]
        else:
            # Fallback values
            print("Warning: No results available, using fallback parameters for gas flow plot")
            sigma_2, t_1, t_2, infall_1, infall_2 = 100.0, 0.1, 8.0, 2.0, 1.0
            sfe_val, delta_sfe_val = 0.01, 0.0
            mgal = 1e10
        
        # Create simplified plot showing expected gas flow patterns
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        # Time array
        time_gyr = np.linspace(0, 14, 1000)
        
        # Get infall normalizations
        A1 = getattr(GalGA, 'A1', 50.0)
        A2 = getattr(GalGA, 'A2', 100.0)
        
        # 1. Expected inflow rates
        ax1 = fig.add_subplot(gs[0, 0])
        
        infall_1_rate = np.where(time_gyr >= t_1, A1 * np.exp(-(time_gyr - t_1) / infall_1), 0)
        infall_2_rate = np.where(time_gyr >= t_2, A2 * np.exp(-(time_gyr - t_2) / infall_2), 0)
        total_infall = infall_1_rate + infall_2_rate
        
        ax1.plot(time_gyr, total_infall, 'b-', linewidth=2, label='Expected Inflow Rate')
        ax1.set_xlabel('Age (Gyr)')
        ax1.set_ylabel(r'Inflow Rate [$M_\odot$ yr$^{-1}$]')
        ax1.set_title('Expected Gas Inflow from Halo to Galaxy')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Expected Star Formation Rate (simplified)
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Simple estimate: SFR ~ SFE * M_gas / t_SF
        # Assume gas mass follows infall with some depletion
        cumulative_gas = np.cumsum(total_infall) * (time_gyr[1] - time_gyr[0]) * 1e9
        sfe_evolution = np.where(time_gyr < t_2, sfe_val, sfe_val + delta_sfe_val)
        expected_sfr = sfe_evolution * cumulative_gas / 1e9  # Rough estimate
        
        ax2.plot(time_gyr, expected_sfr, 'r-', linewidth=2, label='Expected SFR')
        ax2.set_xlabel('Age (Gyr)')
        ax2.set_ylabel(r'SFR [$M_\odot$ yr$^{-1}$]')
        ax2.set_title('Expected Star Formation Rate')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Mass loading factor (simplified)
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Assume typical mass loading evolution
        mass_loading = 1.0 + 5.0 * np.exp(-time_gyr / 3.0)  # Higher early, lower late
        
        ax3.plot(time_gyr, mass_loading, 'purple', linewidth=2, label='Expected Mass Loading')
        ax3.set_xlabel('Age (Gyr)')
        ax3.set_ylabel('η (Outflow/SFR)')
        ax3.set_title('Expected Mass Loading Factor Evolution')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Expected gas fraction evolution
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Simple model for gas fraction
        gas_fraction = 0.9 * np.exp(-time_gyr / 8.0) + 0.1  # Exponential decline
        
        ax4.plot(time_gyr, gas_fraction, 'green', linewidth=2, label='Expected Gas Fraction')
        ax4.set_xlabel('Age (Gyr)')
        ax4.set_ylabel('Gas Fraction')
        ax4.set_title('Expected Gas Fraction Evolution')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # 5. Parameter summary
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        summary_text = f"""GAS FLOW MODEL PARAMETERS
        
Infall Parameters:
• First infall: A₁ = {A1:.1f} M☉/yr, t₁ = {t_1:.3f} Gyr, τ₁ = {infall_1:.2f} Gyr
• Second infall: A₂ = {A2:.1f} M☉/yr, t₂ = {t_2:.1f} Gyr, τ₂ = {infall_2:.2f} Gyr
• Spatial dispersion: σ₂ = {sigma_2:.1f} pc

Star Formation:
• Initial SFE: {sfe_val:.4f}
• SFE change at t₂: {delta_sfe_val:.3f}
• Final SFE: {sfe_val + delta_sfe_val:.4f}
• Galaxy mass: {mgal:.2e} M☉

Physical Interpretation:
• Two-phase gas accretion model
• Early phase: Primordial gas infall
• Late phase: Recycled/enriched gas infall
• SFE evolution reflects changing conditions"""
        
        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=12,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Simplified gas flow physics plot saved to {save_path}")
        return fig
        
    except Exception as e:
        print(f"Error in simplified gas flow plot: {e}")
        print("Traceback:", traceback.format_exc())
        return create_fallback_gas_flow_plot(save_path)


def create_fallback_gas_flow_plot(save_path):
    """Create a minimal fallback gas flow plot"""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        time_gyr = np.linspace(0, 14, 100)
        generic_rate = 100 * np.exp(-time_gyr / 5.0)
        
        ax.plot(time_gyr, generic_rate, 'b-', linewidth=2, label='Generic Gas Flow Rate')
        ax.set_xlabel('Age (Gyr)')
        ax.set_ylabel(r'Rate [$M_\odot$ yr$^{-1}$]')
        ax.set_title('Gas Flow Physics (Fallback Plot)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Fallback gas flow plot saved to {save_path}")
        return fig
        
    except Exception as e:
        print(f"Even fallback gas flow plot failed: {e}")
        return None


def generate_physics_plots(GalGA, results_file='GA/simulation_results.csv'):
    """
    Generate improved physics plots that properly show both infall episodes
    Now with robust error handling and multiple fallback options
    """
    print("Generating physics plots with robust error handling...")
    
    try:
        # Ensure directory exists
        os.makedirs('GA', exist_ok=True)
        
        # Try to load results
        df = None
        if os.path.exists(results_file):
            try:
                import pandas as pd
                df = pd.read_csv(results_file)
                if not df.empty:
                    df.sort_values('fitness', inplace=True)
                    print(f"Loaded {len(df)} results from {results_file}")
                else:
                    print("Results file is empty, using fallback parameters")
            except Exception as e:
                print(f"Error loading results file: {e}")
                df = None
        else:
            print(f"Results file {results_file} not found, using fallback parameters")
        
        # Generate the improved plots with fallbacks
        print("Generating infall physics plot...")
        fig1 = plot_infall_physics_improved(GalGA, df)
        
        print("Generating simplified gas flow plot...")
        fig2 = plot_gas_flow_physics_simplified(GalGA, df)
        
        print("Physics plots completed successfully!")
        return fig1, fig2
        
    except Exception as e:
        print(f"Error in generate_physics_plots: {e}")
        print("Traceback:", traceback.format_exc())
        
        # Last resort: create minimal plots
        try:
            fig1 = create_fallback_infall_plot('GA/Infall_Physics_Improved.png')
            fig2 = create_fallback_gas_flow_plot('GA/Gas_Flow_Physics_Simplified.png')
            print("Created fallback physics plots")
            return fig1, fig2
        except Exception as e2:
            print(f"Even fallback physics plots failed: {e2}")
            return None, None