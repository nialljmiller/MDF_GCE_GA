import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys
sys.path.append('../')
from JINAPyCEE import omega_plus


plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 18,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 1.5,
})


def reconstruct_best_model(GalGA, results_df=None):
    """Reconstruct the omega_plus model for the best-fit parameters"""
    
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
    else:
        r = GalGA.results[0]
        comp_idx, imf_idx, sn1a_idx, sy_idx, sn1ar_idx = int(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4])
        sigma_2, t_1, t_2, infall_1, infall_2 = r[5], r[6], r[7], r[8], r[9]
        sfe_val, delta_sfe_val, imf_upper, mgal, nb = r[10], r[11], r[12], r[13], r[14]
    
    # Get the parameter arrays from GalGA
    comp = GalGA.comp_array[comp_idx]
    imf_val = GalGA.imf_array[imf_idx]
    sn1a = GalGA.sn1a_assumptions[sn1a_idx]
    sy = GalGA.stellar_yield_assumptions[sy_idx]
    sn1ar = GalGA.sn1a_rates[sn1ar_idx]
    
    # Reconstruct the model with the same parameters used in evaluation
    kwargs = {
        'special_timesteps': GalGA.timesteps,
        'twoinfall_sigmas': [1300, sigma_2],
        'galradius': 1800,
        'exp_infall': [[-1, t_1*1e9, infall_1*1e9], [-1, t_2*1e9, infall_2*1e9]],
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

def plot_real_infall_physics(GalGA, results_df=None, save_path='Real_Infall_Physics.png'):
    """
    Generate an enhanced visualization of galactic chemical evolution model physics,
    emphasizing the two-infall paradigm with improved scientific presentation.
    """
    
    save_path = GalGA.output_path + save_path

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Reconstruct the best model
    GCE_model = reconstruct_best_model(GalGA, results_df)
    
    # Extract physical arrays from omega model with proper error handling
    try:
        ages = np.array(GCE_model.inner.history.age) / 1e9
        timesteps_yr = np.array(GCE_model.inner.history.timesteps)
        inflow_masses = np.array(GCE_model.inner.m_inflow_t)
        outflow_masses = np.array(GCE_model.inner.m_outflow_t)
        
        # Ensure consistent array dimensions
        sfr_rates = np.array(GCE_model.inner.history.sfr_abs)[:len(timesteps_yr)]
        metallicity = np.array(GCE_model.inner.history.metallicity)[:len(timesteps_yr)]
        
        # Convert masses to physically meaningful rates
        inflow_rates = inflow_masses / timesteps_yr  # M☉/yr
        outflow_rates = outflow_masses / timesteps_yr  # M☉/yr
        
        # Extract cumulative quantities
        gas_masses = np.array([np.sum(GCE_model.inner.ymgal[i]) for i in range(len(GCE_model.inner.ymgal))])
        stellar_masses_raw = np.array(GCE_model.inner.history.m_locked)
        
        # Ensure stellar mass array matches age array length
        if len(stellar_masses_raw) < len(ages):
            stellar_masses = np.append(stellar_masses_raw, stellar_masses_raw[-1])
        else:
            stellar_masses = stellar_masses_raw[:len(ages)]
            
    except Exception as e:
        print(f"Error extracting model data: {e}")
        return None
    
    # Extract best-fit parameters for physical interpretation
    if results_df is not None and not results_df.empty:
        bm = results_df.iloc[0]
        sigma_2, t_1, t_2 = bm['sigma_2'], bm['t_1'], bm['t_2']
        infall_1, infall_2 = bm['infall_1'], bm['infall_2']
        sfe_val, delta_sfe_val = bm['sfe'], bm['delta_sfe']
        mgal = bm['mgal']
    else:
        r = GalGA.results[0]
        sigma_2, t_1, t_2, infall_1, infall_2 = r[5], r[6], r[7], r[8], r[9]
        sfe_val, delta_sfe_val, mgal = r[10], r[11], r[13]
    
    # Create enhanced figure with improved layout
    fig = plt.figure(figsize=(22, 16))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3, 
                  left=0.06, right=0.98, top=0.94, bottom=0.06)
    

    
    # Define enhanced color palette for scientific clarity
    colors = {
        'inflow': '#1f77b4',      # Professional blue
        'outflow': '#d62728',     # Scientific red  
        'sfr': '#ff7f0e',         # Distinct orange
        'gas': '#2ca02c',         # Scientific green
        'stellar': '#9467bd',     # Professional purple
        'metallicity': '#8c564b',  # Earth tone
        'efficiency': '#e377c2',  # Distinctive pink
        'loading': '#7f7f7f'      # Neutral gray
    }
    
    # ======================================================================
    # PANEL 1: Enhanced Inflow Rate with Theoretical Overlay
    # ======================================================================
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot actual inflow rate with improved styling
    ax1.plot(ages[:-1], inflow_rates, color=colors['inflow'], linewidth=3, 
             label='Computed Inflow Rate', marker='o', markersize=4, alpha=0.9)
    
    # Add theoretical infall episodes with enhanced visualization
    t_theory = np.linspace(0, ages[-1], 1000)
    
    # First infall episode (exponential decay)
    infall_1_theory = np.exp(-t_theory / infall_1) * np.heaviside(t_theory - t_1, 1)
    infall_1_norm = np.max(inflow_rates) * infall_1_theory / np.max(infall_1_theory) if np.max(infall_1_theory) > 0 else infall_1_theory
    
    # Second infall episode  
    infall_2_theory = np.exp(-(t_theory - t_2) / infall_2) * np.heaviside(t_theory - t_2, 1)
    infall_2_norm = np.max(inflow_rates) * 0.3 * infall_2_theory / np.max(infall_2_theory) if np.max(infall_2_theory) > 0 else infall_2_theory
    
    # Plot theoretical curves with transparency
    ax1.plot(t_theory, infall_1_theory, '--', color='lightblue', linewidth=2, alpha=0.7,
             label=f'First Episode (τ={infall_1:.2f} Gyr)')
    ax1.plot(t_theory, infall_2_theory, '--', color='salmon', linewidth=2, alpha=0.7,
             label=f'Second Episode (τ={infall_2:.2f} Gyr)')
    
    # Mark critical epochs
    ax1.axvline(t_1, color='steelblue', linestyle=':', linewidth=2, alpha=0.8)
    ax1.axvline(t_2, color='crimson', linestyle=':', linewidth=2, alpha=0.8)
    
    # Enhanced axis formatting
    ax1.set_xlabel('Time (Gyr)', fontsize=14, fontweight='bold')
    ax1.set_ylabel(r'Inflow Rate ($M_\odot$ yr$^{-1}$)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax1.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax1.set_xlim(0, ages[-1])
    ax1.set_ylim(bottom=0)
    
    # ======================================================================
    # PANEL 2: Star Formation History with Physical Context
    # ======================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.semilogy(ages[:-1], sfr_rates, color=colors['sfr'], linewidth=2.5, 
                 label='SFR', marker='s', markersize=3)
    
    # Highlight SFE change epoch
    ax2.axvline(t_2, color='crimson', linestyle=':', alpha=0.7, linewidth=2)
    ax2.text(t_2 + 0.2, np.max(sfr_rates) * 0.1, f'ΔSFE = {delta_sfe_val:+.4f}', 
             rotation=90, fontsize=10, alpha=0.8, fontweight='bold')
    
    ax2.set_xlabel('Time (Gyr)', fontsize=12, fontweight='bold')
    ax2.set_ylabel(r'SFR ($M_\odot$ yr$^{-1}$)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.2)
    ax2.legend(fontsize=11)
    
    # ======================================================================
    # PANEL 3: Gas Flows and Mass Loading
    # ======================================================================
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(ages[:-1], inflow_rates, color=colors['inflow'], linewidth=2, 
             label='Inflow', marker='o', markersize=2, alpha=0.8)

    ax3.plot(ages[:-1], outflow_rates, color=colors['outflow'], linewidth=2, 
             label='Outflow', marker='^', markersize=2, alpha=0.8)

    ax3.set_xlabel('Time (Gyr)', fontsize=12, fontweight='bold')
    ax3.set_ylabel(r'Flow Rate ($M_\odot$ yr$^{-1}$)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.2)
    ax3.legend(fontsize=11)
    
    # ======================================================================
    # PANEL 4: Mass Loading Factor with Physical Interpretation
    # ======================================================================
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(ages[:-1], inflow_rates, color=colors['inflow'], linewidth=2, 
             label='Inflow', marker='o', markersize=2, alpha=0.8)

    ax4.semilogy(ages[:-1], np.maximum(outflow_rates, 1e-10), 
                 color=colors['outflow'], linewidth=2, 
                 label='Outflow', marker='^', markersize=2, alpha=0.8)


    ax4.set_xlabel('Time (Gyr)', fontsize=12, fontweight='bold')
    ax4.set_ylabel(r'Flow Rate ($M_\odot$ yr$^{-1}$)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.2)
    ax4.legend(fontsize=11)
    
    # ======================================================================
    # PANEL 5: Metallicity Evolution
    # ======================================================================
    ax5 = fig.add_subplot(gs[1, 3])
    ax5.plot(ages[:-1], metallicity, color=colors['metallicity'], linewidth=2.5,
             label='[Fe/H]', marker='o', markersize=3)
    
    ax5.set_xlabel('Time (Gyr)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('[Fe/H]', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.2)
    ax5.legend(fontsize=11)
    
    # ======================================================================
    # PANEL 6: Reservoir Masses with Physical Scaling
    # ======================================================================
    ax6 = fig.add_subplot(gs[2, :2])
    ax6.semilogy(ages, gas_masses, color=colors['gas'], linewidth=3, 
                 label='Gas Reservoir', marker='o', markersize=3, alpha=0.9)
    ax6.semilogy(ages, stellar_masses, color=colors['stellar'], linewidth=3, 
                 label='Stellar Component', marker='s', markersize=3, alpha=0.9)
    
    # Add total mass for context
    total_baryons = gas_masses + stellar_masses
    ax6.semilogy(ages, total_baryons, color='black', linewidth=2, linestyle='--', 
                 label='Total mass', alpha=0.7)
    
    # Mark key transition epochs
    ax6.axvline(t_1, color='steelblue', linestyle=':', alpha=0.6)
    ax6.axvline(t_2, color='crimson', linestyle=':', alpha=0.6)
    
    ax6.set_xlabel('Time (Gyr)', fontsize=12, fontweight='bold')
    ax6.set_ylabel(r'Reservoir Mass ($M_\odot$)', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.2)
    ax6.legend(fontsize=11, loc='best')
    
    # ======================================================================
    # PANEL 7: Gas Fraction and Star Formation Efficiency
    # ======================================================================
    ax7 = fig.add_subplot(gs[2, 2:])
    
    # Primary y-axis: Gas fraction
    gas_fraction = gas_masses / (gas_masses + stellar_masses)
    ax7.plot(ages, gas_fraction, color=colors['gas'], linewidth=3, 
             label='Gas Fraction', marker='v', markersize=3)
    ax7.set_ylabel('Gas Fraction', color=colors['gas'], fontsize=12, fontweight='bold')
    ax7.tick_params(axis='y', labelcolor=colors['gas'])
    
    # Secondary y-axis: Star formation efficiency
    ax7_twin = ax7.twinx()
    # Calculate instantaneous SFE = SFR / M_gas
    sfe_inst = np.where(gas_masses[:-1] > 0, sfr_rates / gas_masses[:-1], 0)
    ax7_twin.semilogy(ages[:-1], sfe_inst, color=colors['efficiency'], linewidth=2, 
                      linestyle='--', label='Instantaneous SFE', marker='x', markersize=3)
    ax7_twin.set_ylabel(r'SFE (yr$^{-1}$)', color=colors['efficiency'], fontsize=12, fontweight='bold')
    ax7_twin.tick_params(axis='y', labelcolor=colors['efficiency'])
    
    ax7.set_xlabel('Time (Gyr)', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.2)
    
    # Combined legend
    lines1, labels1 = ax7.get_legend_handles_labels()
    lines2, labels2 = ax7_twin.get_legend_handles_labels()
    ax7.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=11)
    
    # ======================================================================
    # PANEL 8: Cumulative Budget Analysis
    # ======================================================================
    ax8 = fig.add_subplot(gs[3, :2])
    
    # Cumulative fluxes
    cumulative_inflow = np.cumsum(inflow_masses)
    cumulative_outflow = np.cumsum(outflow_masses)
    cumulative_sf = np.cumsum(sfr_rates * timesteps_yr)
    
    ax8.semilogy(ages[:-1], cumulative_inflow, color=colors['inflow'], linewidth=3, 
                 label='Cumulative Inflow', marker='o', markersize=3)
    ax8.semilogy(ages[:-1], cumulative_sf, color=colors['sfr'], linewidth=3, 
                 label='Cumulative SF', marker='s', markersize=3)
    
    if np.max(cumulative_outflow) > 0:
        ax8.semilogy(ages[:-1], np.maximum(cumulative_outflow, 1e6), color=colors['outflow'], 
                     linewidth=3, label='Cumulative Outflow', marker='^', markersize=3)
    
    ax8.set_xlabel('Time (Gyr)', fontsize=12, fontweight='bold')
    ax8.set_ylabel(r'Cumulative Mass ($M_\odot$)', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.2)
    ax8.legend(fontsize=11)
    

    ax8a = fig.add_subplot(gs[3, 2])
    
    # Cumulative fluxes
    cumulative_inflow = np.cumsum(inflow_masses)
    cumulative_outflow = np.cumsum(outflow_masses)
    cumulative_sf = np.cumsum(sfr_rates * timesteps_yr)
    
    ax8a.plot(ages[:-1], cumulative_inflow, color=colors['inflow'], linewidth=3, 
                 label='Cumulative Inflow', marker='o', markersize=3)
    ax8a.plot(ages[:-1], cumulative_sf, color=colors['sfr'], linewidth=3, 
                 label='Cumulative SF', marker='s', markersize=3)
    
    if np.max(cumulative_outflow) > 0:
        ax8a.plot(ages[:-1], np.maximum(cumulative_outflow, 1e6), color=colors['outflow'], 
                     linewidth=3, label='Cumulative Outflow', marker='^', markersize=3)
    
    ax8a.set_xlabel('Time (Gyr)', fontsize=12, fontweight='bold')
    ax8a.set_ylabel(r'Cumulative Mass ($M_\odot$)', fontsize=12, fontweight='bold')
    ax8a.grid(True, alpha=0.2)
    ax8a.legend(fontsize=11)






    # ======================================================================
    # PANEL 9: Enhanced Physics Summary with Quantitative Analysis
    # ======================================================================
    ax9 = fig.add_subplot(gs[3, 3:])
    ax9.axis('off')
    
    # Calculate physics summary
    total_inflow = np.sum(inflow_masses)
    total_outflow = np.sum(outflow_masses) 
    total_sf = np.sum(sfr_rates * timesteps_yr)
    peak_inflow = np.max(inflow_rates)
    peak_sfr = np.max(sfr_rates)
    final_stellar_mass = stellar_masses[-1]
    final_gas_mass = gas_masses[-1]
    inflow_peak_time = ages[:-1][np.argmax(inflow_rates)]
    
    # Calculate key efficiency metrics
    sf_efficiency = total_sf / total_inflow if total_inflow > 0 else 0
    retention_fraction = (total_inflow - total_outflow) / total_inflow if total_inflow > 0 else 0
    final_gas_fraction = final_gas_mass / (final_gas_mass + final_stellar_mass)
    
    # Determine infall episode characteristics
    if t_2 < 2.0:
        infall_regime = "Early second accretion"
    elif t_2 < 8.0:
        infall_regime = "Intermediate second accretion"  
    else:
        infall_regime = "Late second accretion"
        
    summary_text = f"""PHYSICAL MODEL DIAGNOSTICS

Two-Infall Parameters:
├─ σ₂ = {sigma_2:.1f}
├─ Episode I: t₁ = {t_1:.3f} Gyr, τ₁ = {infall_1:.2f} Gyr
├─ Episode II: t₂ = {t_2:.1f} Gyr, τ₂ = {infall_2:.2f} Gyr  
└─ SFE Evolution: {sfe_val:.4f} → {sfe_val + delta_sfe_val:.4f}

Mass:
├─ Total inflow: {total_inflow:.2e}
├─ Final stellar mass: {final_stellar_mass:.2e}
└─ Final gas reservoir: {final_gas_mass:.2e}

"""
    
    ax9.text(0.02, 0.99, summary_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace', linespacing=1.4,
             bbox=dict(boxstyle="round,pad=0.6", facecolor="lightcyan", 
                      edgecolor="steelblue", alpha=0.95, linewidth=1.5))

    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    plt.close('all') 

    print(f"Enhanced physics diagnostics saved: {save_path}")
    
    return fig

def plot_omega_diagnostics(GalGA, results_df=None, save_path='Omega_Model_Diagnostics.png'):
    """Plot additional diagnostics from the omega model"""
    

    save_path = GalGA.output_path + save_path


    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Reconstruct the best model
    GCE_model = reconstruct_best_model(GalGA, results_df)
    
    # Extract additional omega diagnostics with proper array lengths
    ages = np.array(GCE_model.inner.history.age) / 1e9
    timesteps_yr = np.array(GCE_model.inner.history.timesteps)
    
    # Fix arrays to proper lengths
    metallicity = np.array(GCE_model.inner.history.metallicity)[:len(timesteps_yr)]  # 10 elements
    eta_outflow = np.array(GCE_model.inner.history.eta_outflow_t)[:len(timesteps_yr)]  # 10 elements  
    m_tot_ISM = np.array(GCE_model.inner.history.m_tot_ISM_t)  # Should be 11 elements
    
    # Create diagnostics plot
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Metallicity evolution (10 elements vs 10 elements)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ages[:-1], metallicity, 'gold', linewidth=2, marker='o', markersize=2)
    ax1.set_xlabel('Age (Gyr)')
    ax1.set_ylabel('Metallicity Z')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # 2. Outflow efficiency evolution (10 elements vs 10 elements)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ages[:-1], eta_outflow, 'darkgreen', linewidth=2, marker='s', markersize=2)
    ax2.set_xlabel('Age (Gyr)')
    ax2.set_ylabel('η (Mass Loading)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Total ISM mass (11 elements vs 11 elements)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(ages, m_tot_ISM, 'darkred', linewidth=2, marker='^', markersize=2)
    ax3.set_xlabel('Age (Gyr)')
    ax3.set_ylabel(r'Total ISM Mass [$M_\odot$]')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # 4. Halo properties (11 elements vs 11 elements)
    ax4 = fig.add_subplot(gs[1, :])
    halo_masses = [np.sum(outer) for outer in GCE_model.ymgal_outer]
    ax4.plot(ages, halo_masses, 'purple', linewidth=2, marker='d', markersize=2, label='Halo Gas Mass')
    ax4.set_xlabel('Age (Gyr)')
    ax4.set_ylabel(r'Halo Gas Mass [$M_\odot$]')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Omega diagnostics plot saved to {save_path}")
    return fig


def generate_physics_plots(GalGA, results_file='GA/simulation_results.csv'):
    """Generate physics plots using actual omega model computations"""
    
    print("Generating physics plots using actual omega model data...")
    
    os.makedirs('GA', exist_ok=True)
    
    # Load results
    import pandas as pd
    df = pd.read_csv(results_file)
    df.sort_values('fitness', inplace=True)
    print(f"Loaded {len(df)} results from {results_file}")
    
    # Generate the physics plots using real omega data
    print("Generating real infall physics plot from omega model...")
    fig1 = plot_real_infall_physics(GalGA, df)
    
    print("Generating omega model diagnostics...")
    fig2 = plot_omega_diagnostics(GalGA, df)
    
    print("Physics plots using omega model data completed!")
    plt.close('all')               # (optional) belt-and-suspenders at the end of an iteration
    return fig1, fig2