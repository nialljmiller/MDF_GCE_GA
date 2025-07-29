import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

def plot_infall_physics_improved(GalGA, results_df=None, save_path='GA/Infall_Physics_Improved.png'):
    """
    Improved plot of the two-infall model physics showing both gas accretion episodes clearly
    """
    # Get best model parameters
    if results_df is not None and not results_df.empty:
        bm = results_df.iloc[0]
        best_params = (bm['sigma_2'], bm['t_2'], bm['infall_2'])
        sigma_2, t_1, t_2, infall_1, infall_2 = bm['sigma_2'], bm['t_1'], bm['t_2'], bm['infall_1'], bm['infall_2']
        sfe_val, delta_sfe_val = bm['sfe'], bm['delta_sfe']
    else:
        r = GalGA.results[0]
        best_params = (r[5], r[7], r[9])
        sigma_2, t_1, t_2, infall_1, infall_2 = r[5], r[6], r[7], r[8], r[9]
        sfe_val, delta_sfe_val = r[10], r[11]
    
    # Create the plot
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    # Time array for infall functions - extend to show full evolution
    time_gyr = np.linspace(0, 14, 2000)
    
    # Get the actual A1 and A2 values from the GalGA object
    A1 = GalGA.A1 if hasattr(GalGA, 'A1') else 50.0
    A2 = GalGA.A2 if hasattr(GalGA, 'A2') else 100.0
    
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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Improved infall physics plot saved to {save_path}")
    print(f"First infall: peak = {first_infall_peak:.1f} M☉/yr, total = {first_infall_total:.2e} M☉")
    print(f"Second infall: peak = {second_infall_peak:.1f} M☉/yr, total = {second_infall_total:.2e} M☉")
    
    return fig


def plot_gas_flow_physics_improved(GalGA, results_df=None, save_path='GA/Gas_Flow_Physics_Improved.png'):
    """
    Enhanced plot of gas flow physics with better visualization of the two-zone model
    """
    # Get best model parameters and reconstruct the model
    if results_df is not None and not results_df.empty:
        bm = results_df.iloc[0]
        best_params = (bm['sigma_2'], bm['t_2'], bm['infall_2'])
    else:
        r = GalGA.results[0]
        best_params = (r[5], r[7], r[9])

    # Find and reconstruct the best model (same as before)
    best_model = None
    for label, res in zip(GalGA.labels, GalGA.results):
        params = (res[5], res[7], res[9])
        if all(abs(p - b) < 1e-5 for p, b in zip(params, best_params)):
            # Reconstruct model parameters
            comp_idx, imf_idx, sn1a_idx, sy_idx, sn1ar_idx = res[0], res[1], res[2], res[3], res[4]
            sigma_2, t_1, t_2, infall_1, infall_2 = res[5], res[6], res[7], res[8], res[9]
            sfe_val, delta_sfe_val, imf_upper, mgal, nb = res[10], res[11], res[12], res[13], res[14]
            
            comp = GalGA.comp_array[comp_idx]
            imf_val = GalGA.imf_array[imf_idx]
            sn1a = GalGA.sn1a_assumptions[sn1a_idx]
            sy = GalGA.stellar_yield_assumptions[sy_idx]
            sn1ar = GalGA.sn1a_rates[sn1ar_idx]
            
            from JINAPyCEE import omega_plus
            
            kwargs = {
                'special_timesteps': GalGA.timesteps,
                'twoinfall_sigmas': [1300, sigma_2],
                'galradius': 1800,
                'exp_infall': [[GalGA.A1, t_1*1e9, infall_1*1e9], [GalGA.A2, t_2*1e9, infall_2*1e9]],            
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
            
            best_model = omega_plus.omega_plus(**kwargs)
            break
    
    if best_model is None:
        print("Could not reconstruct best model for gas flow physics plot")
        return None
    
    # Create enhanced plot
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    # Get time arrays with proper handling of dimension mismatches
    age_gyr = np.array(best_model.inner.history.age) / 1e9
    timesteps = np.array(best_model.inner.history.timesteps) / 1e9
    
    # Ensure arrays match properly
    min_len = min(len(age_gyr)-1, len(timesteps))
    age_gyr_timesteps = age_gyr[:min_len]
    
    # Get flow data
    if hasattr(best_model.inner, 'm_inflow_t') and len(best_model.inner.m_inflow_t) > 0:
        inflow_rates = np.array(best_model.inner.m_inflow_t[:min_len]) / timesteps[:min_len]
        outflow_rates = np.array(best_model.inner.m_outflow_t[:min_len]) / timesteps[:min_len]
        eta_outflow = np.array(best_model.inner.eta_outflow_t[:min_len])
        sfr = np.array(best_model.inner.history.sfr_abs[:min_len])
    else:
        # Fallback to zeros if data not available
        inflow_rates = np.zeros(min_len)
        outflow_rates = np.zeros(min_len)
        eta_outflow = np.zeros(min_len)
        sfr = np.array(best_model.inner.history.sfr_abs[:min_len])
    
    # 1. Inflow rates over time with better scaling
    ax1 = fig.add_subplot(gs[0, 0])
    mask_inflow = inflow_rates > 0
    if np.any(mask_inflow):
        ax1.plot(age_gyr_timesteps[mask_inflow], inflow_rates[mask_inflow], 'b-', linewidth=2, label='Galactic Inflow')
        ax1.set_yscale('log')
    else:
        ax1.plot(age_gyr_timesteps, inflow_rates, 'b-', linewidth=2, label='Galactic Inflow')
    
    ax1.set_xlabel('Age (Gyr)')
    ax1.set_ylabel(r'Inflow Rate [$M_\odot$ yr$^{-1}$]')
    ax1.set_title('Gas Inflow from Halo to Galaxy')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Star Formation Rate
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(age_gyr_timesteps, sfr, 'r-', linewidth=2, label='Star Formation Rate')
    ax2.set_xlabel('Age (Gyr)')
    ax2.set_ylabel(r'SFR [$M_\odot$ yr$^{-1}$]')
    ax2.set_title('Star Formation Rate')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Outflow rates
    ax3 = fig.add_subplot(gs[0, 2])
    mask_outflow = outflow_rates > 0
    if np.any(mask_outflow):
        ax3.plot(age_gyr_timesteps[mask_outflow], outflow_rates[mask_outflow], 'g-', linewidth=2, label='Galactic Outflow')
        ax3.set_yscale('log')
    else:
        ax3.plot(age_gyr_timesteps, outflow_rates, 'g-', linewidth=2, label='Galactic Outflow')
    
    ax3.set_xlabel('Age (Gyr)')
    ax3.set_ylabel(r'Outflow Rate [$M_\odot$ yr$^{-1}$]')
    ax3.set_title('Gas Outflow from Galaxy')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Mass loading factor evolution
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(age_gyr_timesteps, eta_outflow, 'purple', linewidth=2, label='Mass Loading')
    ax4.set_xlabel('Age (Gyr)')
    ax4.set_ylabel('η (Outflow/SFR)')
    ax4.set_title('Mass Loading Factor Evolution')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 5. Gas masses in both zones (improved)
    ax5 = fig.add_subplot(gs[1, 1:])
    
    # Calculate masses more carefully
    max_time_len = len(age_gyr)
    inner_gas_mass = np.array([np.sum(best_model.inner.ymgal[i]) for i in range(min(max_time_len, len(best_model.inner.ymgal)))])
    outer_gas_mass = np.array([np.sum(best_model.ymgal_outer[i]) for i in range(min(max_time_len, len(best_model.ymgal_outer)))])
    stellar_mass = np.array(best_model.inner.history.m_locked)
    
    # Ensure all arrays have the same length
    min_mass_len = min(len(age_gyr), len(inner_gas_mass), len(outer_gas_mass), len(stellar_mass))
    age_gyr_mass = age_gyr[:min_mass_len]
    inner_gas_mass = inner_gas_mass[:min_mass_len]
    outer_gas_mass = outer_gas_mass[:min_mass_len]
    stellar_mass = stellar_mass[:min_mass_len]
    
    ax5.plot(age_gyr_mass, inner_gas_mass, 'b-', linewidth=2, label='Inner Gas Mass (Galaxy)')
    ax5.plot(age_gyr_mass, outer_gas_mass, 'orange', linewidth=2, label='Outer Gas Mass (Halo)')
    ax5.plot(age_gyr_mass, stellar_mass, 'r-', linewidth=2, label='Stellar Mass')
    
    ax5.set_xlabel('Age (Gyr)')
    ax5.set_ylabel(r'Mass [$M_\odot$]')
    ax5.set_title('Mass Evolution in Two-Zone Model')
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 6. Flow balance analysis
    ax6 = fig.add_subplot(gs[2, :])
    
    # Calculate net flow (inflow - outflow)
    net_flow = inflow_rates - outflow_rates
    
    ax6.fill_between(age_gyr_timesteps, 0, inflow_rates, alpha=0.3, color='blue', label='Inflow Rate')
    ax6.fill_between(age_gyr_timesteps, 0, -outflow_rates, alpha=0.3, color='red', label='Outflow Rate')
    ax6.plot(age_gyr_timesteps, net_flow, 'black', linewidth=2, label='Net Flow (Inflow - Outflow)')
    ax6.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    ax6.set_xlabel('Age (Gyr)')
    ax6.set_ylabel(r'Flow Rate [$M_\odot$ yr$^{-1}$]')
    ax6.set_title('Gas Flow Balance Analysis')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # 7. Gas fraction evolution
    ax7 = fig.add_subplot(gs[3, 0])
    
    total_mass = inner_gas_mass + outer_gas_mass + stellar_mass
    gas_fraction = (inner_gas_mass + outer_gas_mass) / total_mass
    inner_gas_fraction = inner_gas_mass / total_mass
    
    ax7.plot(age_gyr_mass, gas_fraction, 'blue', linewidth=2, label='Total Gas Fraction')
    ax7.plot(age_gyr_mass, inner_gas_fraction, 'green', linewidth=2, label='Inner Gas Fraction')
    
    ax7.set_xlabel('Age (Gyr)')
    ax7.set_ylabel('Gas Fraction')
    ax7.set_title('Gas Fraction Evolution')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    # 8. Summary statistics
    ax8 = fig.add_subplot(gs[3, 1:])
    ax8.axis('off')
    
    # Calculate key statistics
    total_inflow = np.trapz(inflow_rates[inflow_rates > 0], age_gyr_timesteps[inflow_rates > 0]) if np.any(inflow_rates > 0) else 0
    total_outflow = np.trapz(outflow_rates[outflow_rates > 0], age_gyr_timesteps[outflow_rates > 0]) if np.any(outflow_rates > 0) else 0
    total_sf = np.trapz(sfr, age_gyr_timesteps)
    final_stellar_mass = stellar_mass[-1] if len(stellar_mass) > 0 else 0
    final_gas_mass = (inner_gas_mass[-1] + outer_gas_mass[-1]) if len(inner_gas_mass) > 0 else 0
    
    summary_text = f"""GAS FLOW SUMMARY
    
Total Integrated Flows:
• Total inflow: {total_inflow:.2e} M☉
• Total outflow: {total_outflow:.2e} M☉
• Net flow: {total_inflow - total_outflow:.2e} M☉
• Total star formation: {total_sf:.2e} M☉

Final Masses:
• Stellar mass: {final_stellar_mass:.2e} M☉
• Gas mass: {final_gas_mass:.2e} M☉
• Total baryonic: {final_stellar_mass + final_gas_mass:.2e} M☉

Efficiencies:
• Star formation efficiency: {final_stellar_mass / (total_inflow + 1e-10):.3f}
• Gas retention: {final_gas_mass / (total_inflow + 1e-10):.3f}
• Outflow efficiency: {total_outflow / (total_inflow + 1e-10):.3f}"""
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Improved gas flow physics plot saved to {save_path}")
    return fig


def generate_physics_plots(GalGA, results_file='GA/simulation_results.csv'):
    """
    Generate improved physics plots that properly show both infall episodes
    """
    import pandas as pd
    
    # Load results
    df = pd.read_csv(results_file)
    df.sort_values('fitness', inplace=True)
    
    print("Generating improved physics plots...")
    
    # Generate the improved plots
    fig1 = plot_infall_physics_improved(GalGA, df)
    fig2 = plot_gas_flow_physics_improved(GalGA, df)
    
    print("Improved physics plots completed!")
    return fig1, fig2