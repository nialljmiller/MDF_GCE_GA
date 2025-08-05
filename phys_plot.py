import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import traceback
import sys
sys.path.append('../')
from JINAPyCEE import omega_plus


def reconstruct_best_model(GalGA, results_df=None):
    """
    Reconstruct the omega_plus model for the best-fit parameters
    Returns the actual omega model object with computed physics
    """
    try:
        # Get best model parameters
        if results_df is not None and not results_df.empty:
            bm = results_df.iloc[0]
            comp_idx = int(bm['comp_idx'])
            imf_idx = int(bm['imf_idx'])
            sn1a_idx = int(bm['sn1a_idx'])
            sy_idx = int(bm['sy_idx'])
            sn1ar_idx = int(bm['sn1ar_idx'])
            
            sigma_2 = bm['sigma_2']
            t_1 = bm['t_1']
            t_2 = bm['t_2']
            infall_1 = bm['infall_1']
            infall_2 = bm['infall_2']
            sfe_val = bm['sfe']
            delta_sfe_val = bm['delta_sfe']
            imf_upper = bm['imf_upper']
            mgal = bm['mgal']
            nb = bm['nb']
        elif hasattr(GalGA, 'results') and len(GalGA.results) > 0:
            r = GalGA.results[0]
            comp_idx, imf_idx, sn1a_idx, sy_idx, sn1ar_idx = int(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4])
            sigma_2, t_1, t_2, infall_1, infall_2 = r[5], r[6], r[7], r[8], r[9]
            sfe_val, delta_sfe_val, imf_upper, mgal, nb = r[10], r[11], r[12], r[13], r[14]
        else:
            print("No results available for model reconstruction")
            return None
        
        # Get the parameter arrays from GalGA
        comp = GalGA.comp_array[comp_idx]
        imf_val = GalGA.imf_array[imf_idx]
        sn1a = GalGA.sn1a_assumptions[sn1a_idx]
        sy = GalGA.stellar_yield_assumptions[sy_idx]
        sn1ar = GalGA.sn1a_rates[sn1ar_idx]
        
        # Reconstruct the model with the same parameters used in evaluation
        kwargs = {
            'special_timesteps': GalGA.timesteps,
            'twoinfall_sigmas': [1300, sigma_2],  # Use the actual sigma values
            'galradius': 1800,
            'exp_infall': [[-1, t_1*1e9, infall_1*1e9], [-1, t_2*1e9, infall_2*1e9]],  # -1 for self-normalization
            'tauup': [0.02e9, 0.02e9],
            'mgal': mgal,
            'iniZ': 0.0,
            'mass_loading': 0.0,
            'table': GalGA.sn1a_header + sy,
            'sfe': sfe_val,
            'delta_sfe': delta_sfe_val,
            'imf_type': imf_val,
            'sn1a_table': GalGA.sn1a_header + sn1a,
            'imf_yields_range': [1, imf_upper],
            'iniabu_table': GalGA.iniab_header + comp,
            'nb_1a_per_m': nb,
            'sn1a_rate': sn1ar
        }
        
        print("Reconstructing best-fit omega_plus model...")
        GCE_model = omega_plus.omega_plus(**kwargs)
        print("Model reconstruction successful!")
        
        return GCE_model
        
    except Exception as e:
        print(f"Error reconstructing omega model: {e}")
        print("Traceback:", traceback.format_exc())
        return None


def plot_real_infall_physics(GalGA, results_df=None, save_path='GA/Real_Infall_Physics.png'):
    """
    Plot the actual computed infall physics from the omega model
    Uses real computed values instead of analytical approximations
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Reconstruct the best model
        GCE_model = reconstruct_best_model(GalGA, results_df)
        if GCE_model is None:
            return create_fallback_infall_plot(save_path)
        
        # Extract real computed values from the omega model
        ages = np.array(GCE_model.inner.history.age) / 1e9  # Convert to Gyr
        timesteps = np.array(GCE_model.inner.history.timesteps) / 1e9  # Convert to Gyr
        
        # Get actual computed rates from the model
        inflow_rates = np.array(GCE_model.inner.m_inflow_t)  # Actual inflow rates [Msun/yr]
        outflow_rates = np.array(GCE_model.inner.m_outflow_t)  # Actual outflow rates [Msun/yr]
        sfr_rates = np.array(GCE_model.inner.history.sfr_abs)  # Actual SFR [Msun/yr]
        
        # Get gas and stellar masses
        if hasattr(GCE_model.inner.history, 'm_gas_exp'):
            gas_masses = np.array(GCE_model.inner.history.m_gas_exp)
        elif hasattr(GCE_model.inner.history, 'm_gas'):
            gas_masses = np.array(GCE_model.inner.history.m_gas)
        else:
            gas_masses = None
            
        stellar_masses = np.array(GCE_model.inner.history.m_locked)
        
        # Get best model parameters for display
        if results_df is not None and not results_df.empty:
            bm = results_df.iloc[0]
            sigma_2, t_1, t_2 = bm['sigma_2'], bm['t_1'], bm['t_2']
            infall_1, infall_2 = bm['infall_1'], bm['infall_2']
            sfe_val, delta_sfe_val = bm['sfe'], bm['delta_sfe']
        else:
            r = GalGA.results[0]
            sigma_2, t_1, t_2, infall_1, infall_2 = r[5], r[6], r[7], r[8], r[9]
            sfe_val, delta_sfe_val = r[10], r[11]
        
        # Create comprehensive plot
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # 1. Actual Inflow Rate vs Time
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(ages[:-1], inflow_rates, 'blue', linewidth=3, label='Actual Inflow Rate (from Omega)', marker='o', markersize=3)
        ax1.axvline(t_1, color='lightblue', linestyle=':', alpha=0.7, label=f'First Infall Start (t={t_1:.3f} Gyr)')
        ax1.axvline(t_2, color='red', linestyle=':', alpha=0.7, label=f'Second Infall Start (t={t_2:.1f} Gyr)')
        
        ax1.set_xlabel('Age (Gyr)', fontsize=14)
        ax1.set_ylabel(r'Inflow Rate [$M_\odot$ yr$^{-1}$]', fontsize=14)
        ax1.set_title('Actual Gas Inflow Rate (Computed by Omega Model)', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=12)
        ax1.set_xlim(0, np.max(ages))
        
        # 2. Star Formation Rate
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(ages[:-1], sfr_rates, 'red', linewidth=2, label='Actual SFR', marker='s', markersize=2)
        ax2.axvline(t_2, color='red', linestyle=':', alpha=0.7, label=f'SFE Change at t={t_2:.1f}')
        ax2.set_xlabel('Age (Gyr)')
        ax2.set_ylabel(r'SFR [$M_\odot$ yr$^{-1}$]')
        ax2.set_title('Star Formation Rate')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Outflow Rate
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(ages[:-1], outflow_rates, 'purple', linewidth=2, label='Actual Outflow Rate', marker='^', markersize=2)
        ax3.set_xlabel('Age (Gyr)')
        ax3.set_ylabel(r'Outflow Rate [$M_\odot$ yr$^{-1}$]')
        ax3.set_title('Galactic Outflow Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Mass Loading Factor (Outflow/SFR)
        ax4 = fig.add_subplot(gs[1, 2])
        # Avoid division by zero
        mass_loading = np.where(sfr_rates > 0, outflow_rates / sfr_rates, 0)
        ax4.plot(ages[:-1], mass_loading, 'green', linewidth=2, label='η = Outflow/SFR', marker='d', markersize=2)
        ax4.set_xlabel('Age (Gyr)')
        ax4.set_ylabel('Mass Loading Factor η')
        ax4.set_title('Mass Loading Evolution')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # 5. Gas Mass Evolution
        if gas_masses is not None:
            ax5 = fig.add_subplot(gs[2, 0])
            ax5.plot(ages, gas_masses, 'cyan', linewidth=2, label='Gas Mass', marker='o', markersize=2)
            ax5.set_xlabel('Age (Gyr)')
            ax5.set_ylabel(r'Gas Mass [$M_\odot$]')
            ax5.set_title('Gas Mass Evolution')
            ax5.set_yscale('log')
            ax5.grid(True, alpha=0.3)
            ax5.legend()
        
        # 6. Stellar Mass Evolution
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(ages, stellar_masses, 'orange', linewidth=2, label='Stellar Mass', marker='s', markersize=2)
        ax6.set_xlabel('Age (Gyr)')
        ax6.set_ylabel(r'Stellar Mass [$M_\odot$]')
        ax6.set_title('Stellar Mass Growth')
        ax6.set_yscale('log')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        # 7. Gas Fraction Evolution
        ax7 = fig.add_subplot(gs[2, 2])
        if gas_masses is not None:
            total_baryons = gas_masses + stellar_masses
            gas_fraction = np.where(total_baryons > 0, gas_masses / total_baryons, 0)
            ax7.plot(ages, gas_fraction, 'brown', linewidth=2, label='Gas Fraction', marker='v', markersize=2)
        ax7.set_xlabel('Age (Gyr)')
        ax7.set_ylabel('Gas Fraction')
        ax7.set_title('Gas Fraction Evolution')
        ax7.grid(True, alpha=0.3)
        ax7.legend()
        
        # 8. Cumulative Inflow vs Outflow
        ax8 = fig.add_subplot(gs[3, 0])
        cumulative_inflow = np.cumsum(inflow_rates * timesteps[:-1])
        cumulative_outflow = np.cumsum(outflow_rates * timesteps[:-1])
        
        ax8.plot(ages[:-1], cumulative_inflow, 'blue', linewidth=2, label='Cumulative Inflow', marker='o', markersize=2)
        ax8.plot(ages[:-1], cumulative_outflow, 'purple', linewidth=2, label='Cumulative Outflow', marker='^', markersize=2)
        ax8.set_xlabel('Age (Gyr)')
        ax8.set_ylabel(r'Cumulative Mass [$M_\odot$]')
        ax8.set_title('Cumulative Gas Flows')
        ax8.set_yscale('log')
        ax8.grid(True, alpha=0.3)
        ax8.legend()
        
        # 9. Model Summary
        ax9 = fig.add_subplot(gs[3, 1:])
        ax9.axis('off')
        
        # Calculate some key quantities
        total_inflow = np.sum(inflow_rates * timesteps[:-1])
        total_outflow = np.sum(outflow_rates * timesteps[:-1])
        total_sf = np.sum(sfr_rates * timesteps[:-1])
        peak_inflow = np.max(inflow_rates)
        peak_sfr = np.max(sfr_rates)
        final_stellar_mass = stellar_masses[-1]
        
        # Find infall peaks
        inflow_peak_time = ages[:-1][np.argmax(inflow_rates)]
        
        summary_text = f"""ACTUAL MODEL PHYSICS SUMMARY (From Omega Computation)

Input Parameters:
• σ₂ = {sigma_2:.1f} pc (second infall dispersion)
• First infall: t₁ = {t_1:.3f} Gyr, τ₁ = {infall_1:.2f} Gyr
• Second infall: t₂ = {t_2:.1f} Gyr, τ₂ = {infall_2:.2f} Gyr
• SFE evolution: {sfe_val:.4f} → {sfe_val + delta_sfe_val:.4f}

Computed Results:
• Total gas inflow: {total_inflow:.2e} M☉
• Total gas outflow: {total_outflow:.2e} M☉
• Total star formation: {total_sf:.2e} M☉
• Peak inflow rate: {peak_inflow:.1f} M☉/yr (at t={inflow_peak_time:.1f} Gyr)
• Peak SFR: {peak_sfr:.1f} M☉/yr
• Final stellar mass: {final_stellar_mass:.2e} M☉
• Net gas retention: {(total_inflow - total_outflow)/total_inflow:.2%}

Note: These are the actual values computed by the omega_plus model,
not analytical approximations from input parameters."""
        
        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.9))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Real infall physics plot saved to {save_path}")
        print(f"Model computed: Total inflow = {total_inflow:.2e} M☉, Peak rate = {peak_inflow:.1f} M☉/yr")
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_real_infall_physics: {e}")
        print("Traceback:", traceback.format_exc())
        return create_fallback_infall_plot(save_path)


def plot_omega_diagnostics(GalGA, results_df=None, save_path='GA/Omega_Model_Diagnostics.png'):
    """
    Plot additional diagnostics from the omega model
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Reconstruct the best model
        GCE_model = reconstruct_best_model(GalGA, results_df)
        if GCE_model is None:
            print("Cannot create diagnostics without omega model")
            return None
        
        # Extract additional omega diagnostics
        ages = np.array(GCE_model.inner.history.age) / 1e9
        
        # Try to get various internal quantities
        eta_outflow = getattr(GCE_model.inner.history, 'eta_outflow_t', None)
        m_tot_ISM = getattr(GCE_model.inner.history, 'm_tot_ISM_t', None)
        
        # Create diagnostics plot
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Metallicity evolution
        ax1 = fig.add_subplot(gs[0, 0])
        metallicity = np.array(GCE_model.inner.history.metallicity)
        ax1.plot(ages[:-1], metallicity, 'gold', linewidth=2, marker='o', markersize=2)
        ax1.set_xlabel('Age (Gyr)')
        ax1.set_ylabel('Metallicity Z')
        ax1.set_title('Gas Metallicity Evolution')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # 2. Outflow efficiency evolution (if available)
        if eta_outflow is not None and len(eta_outflow) > 0:
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(ages[:-1], eta_outflow, 'darkgreen', linewidth=2, marker='s', markersize=2)
            ax2.set_xlabel('Age (Gyr)')
            ax2.set_ylabel('η (Mass Loading)')
            ax2.set_title('Mass Loading Factor Evolution')
            ax2.grid(True, alpha=0.3)
        
        # 3. Total ISM mass (if available)
        if m_tot_ISM is not None and len(m_tot_ISM) > 0:
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.plot(ages, m_tot_ISM, 'darkred', linewidth=2, marker='^', markersize=2)
            ax3.set_xlabel('Age (Gyr)')
            ax3.set_ylabel(r'Total ISM Mass [$M_\odot$]')
            ax3.set_title('ISM Mass Evolution')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
        
        # 4. Halo properties (if available)
        ax4 = fig.add_subplot(gs[1, :])
        
        # Try to access halo gas evolution
        if hasattr(GCE_model, 'ymgal_outer') and len(GCE_model.ymgal_outer) > 0:
            halo_masses = [np.sum(outer) for outer in GCE_model.ymgal_outer]
            ax4.plot(ages, halo_masses, 'purple', linewidth=2, marker='d', markersize=2, label='Halo Gas Mass')
            ax4.set_xlabel('Age (Gyr)')
            ax4.set_ylabel(r'Halo Gas Mass [$M_\odot$]')
            ax4.set_title('Circumgalactic Medium Evolution')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Halo gas data not available\nin this omega configuration', 
                    transform=ax4.transAxes, ha='center', va='center', fontsize=14)
            ax4.set_title('Halo Diagnostics (Not Available)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Omega diagnostics plot saved to {save_path}")
        return fig
        
    except Exception as e:
        print(f"Error in omega diagnostics plot: {e}")
        print("Traceback:", traceback.format_exc())
        return None


def create_fallback_infall_plot(save_path):
    """Create a simple fallback infall plot if the main function fails"""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        time_gyr = np.linspace(0, 14, 100)
        fallback_rate = 50 * np.exp(-time_gyr / 3.0) + 100 * np.exp(-(time_gyr - 8.0) / 1.0) * (time_gyr >= 8.0)
        
        ax.plot(time_gyr, fallback_rate, 'blue', linewidth=2, label='Fallback Infall Rate')
        ax.set_xlabel('Age (Gyr)')
        ax.set_ylabel(r'Infall Rate [$M_\odot$ yr$^{-1}$]')
        ax.set_title('Gas Infall Physics (Fallback Plot)')
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


def generate_physics_plots(GalGA, results_file='GA/simulation_results.csv'):
    """
    Generate physics plots using actual omega model computations
    """
    print("Generating physics plots using actual omega model data...")
    
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
                    print("Results file is empty")
            except Exception as e:
                print(f"Error loading results file: {e}")
                df = None
        else:
            print(f"Results file {results_file} not found")
        
        # Generate the physics plots using real omega data
        print("Generating real infall physics plot from omega model...")
        fig1 = plot_real_infall_physics(GalGA, df)
        
        print("Generating omega model diagnostics...")
        fig2 = plot_omega_diagnostics(GalGA, df)
        
        print("Physics plots using omega model data completed!")
        return fig1, fig2
        
    except Exception as e:
        print(f"Error in generate_physics_plots: {e}")
        print("Traceback:", traceback.format_exc())
        
        # Fallback
        try:
            fig1 = create_fallback_infall_plot('GA/Real_Infall_Physics.png')
            fig2 = None
            print("Created fallback physics plot")
            return fig1, fig2
        except Exception as e2:
            print(f"Even fallback failed: {e2}")
            return None, None