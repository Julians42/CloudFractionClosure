using Glob
using NCDatasets
using Statistics
using CairoMakie

import ClimaAtmos as CA 
import ClimaAtmos.Parameters as CAP
import Thermodynamics as TD

using Roots # root solver for theta_li computation

# Path to LES output data
LES_DATA_PATH = "/net/sampo/data1/zhaoyi/GCMForcedLES/cfsite/07/HadGEM2-A/amip/Output.cfsite17_HadGEM2-A_amip_2004-2008.07.4x/stats/Stats.cfsite17_HadGEM2-A_amip_2004-2008.07.nc"

# Load LES datasets
ds_profiles = NCDataset(LES_DATA_PATH).group["profiles"]
ds_timeseries = NCDataset(LES_DATA_PATH).group["timeseries"]

# Initialize ClimaAtmos configuration and parameters
config = CA.AtmosConfig("ClimaAtmos.jl/config/model_configs/prognostic_edmfx_bomex_pigroup_column.yml")
params = CA.ClimaAtmosParameters(config)
CAP.turbconv_params(params)

"""
    tke(ds)

Compute turbulent kinetic energy from velocity variances.
"""
function tke(ds)
    return (ds["w_mean2"][:, :] .+ ds["v_mean2"][:, :] .+ ds["u_mean2"][:, :]) ./ 2
end

"""
    add_z_dim(ds, varname, ds_profs=ds_profiles)

Extend a time series variable to all heights by repeating.
"""
function add_z_dim(ds, varname, ds_profs=ds_profiles)
    repeat(ds[varname][:]', ds_profs.dim["z"], 1)
end

"""
    add_t_dim(ds, varname, ds_profs=ds_profiles)

Extend a profile variable to all times by repeating.
"""
function add_t_dim(ds, varname, ds_profs=ds_profiles)
    repeat(ds[varname][:], 1, ds_profs.dim["t"])
end



"""
    compute_mixing_length(ds_profiles, ds_timeseries, params)

Compute the mixing length and its components from LES data.

# Arguments
- `ds_profiles`: Dataset containing vertical profiles
- `ds_timeseries`: Dataset containing time series data
- `params`: ClimaAtmos parameters

# Returns
- `mix_len`: Full mixing length object containing all components
- `mix_len_dz_indep`: Minimum of TKE, wall, and buoyancy components
"""
function compute_mixing_length(ds_profiles, ds_timeseries, params)
    # Get friction velocity and extend to all heights
    ustar = add_z_dim(ds_timeseries, "friction_velocity_mean")
    
    # Get height coordinate and extend to all times
    ᶜz = add_t_dim(ds_profiles, "z")
    z_sfc = 0  # Surface elevation over ocean
    ᶜdz = ᶜz[2] - ᶜz[1]  # Grid spacing
    
    # Compute surface TKE and extend to all heights
    sfc_tke = repeat(tke(ds_profiles)[1, :]', ds_profiles.dim["z"], 1)
    
    # Get buoyancy and strain rate fields
    ᶜlinear_buoygrad = ds_profiles["buoyancy_frequency_mean2"][:, :]
    ᶜstrain_rate_norm = ds_profiles["strain_rate_magnitude"][:, :]
    
    # Compute TKE and turbulent Prandtl number
    ᶜtke = tke(ds_profiles)
    ᶜPr = CA.turbulent_prandtl_number.(
        params,
        ds_profiles["buoyancy_frequency_mean2"][:, :],
        ds_profiles["strain_rate_magnitude"][:, :]
    )
    
    # Get Obukhov length and extend to all heights
    obukhov_length = add_z_dim(ds_timeseries, "obukhov_length_mean")
    
    # Initialize TKE exchange term
    ᶜtke_exch = zeros(size(ᶜz))
    
    # Compute full mixing length object
    mix_len = CA.mixing_length.(
        params,
        ustar,
        ᶜz,
        z_sfc,
        ᶜdz,
        sfc_tke,
        ᶜlinear_buoygrad,
        ᶜtke,
        obukhov_length,
        ᶜstrain_rate_norm,
        ᶜPr,
        zeros(size(ᶜtke_exch)),
        CA.SmoothMinimumBlending(),
    )
    
    # Compute minimum of TKE, wall, and buoyancy components
    mix_len_dz_indep = [minimum([m.tke, m.wall, m.buoy]) for m in mix_len]
    
    return mix_len, mix_len_dz_indep
end

"""
    compute_vertical_gradient(profile, z)

Compute the vertical gradient of a profile variable.

# Arguments
- `profile`: 2D array of shape (t, z) containing the profile data
- `z`: 1D array of heights in meters

# Returns
- `gradient`: 2D array of shape (t, z) containing the vertical gradient
"""
function compute_vertical_gradient(profile, z)
    # Initialize output array with same size as input
    gradient = similar(profile)
    
    # For each time step
    for t in axes(profile, 1)
        # Compute centered differences for interior points
        for i in 2:(length(z)-1)
            dz = z[i+1] - z[i-1]
            gradient[t, i] = (profile[t, i+1] - profile[t, i-1]) / dz
        end
        
        # Use forward/backward differences at boundaries
        gradient[t, 1] = (profile[t, 2] - profile[t, 1]) / (z[2] - z[1])
        gradient[t, end] = (profile[t, end] - profile[t, end-1]) / (z[end] - z[end-1])
    end
    
    return gradient
end


"""
    objective(θ_li)

Objective function for finding condensation point. Returns 1.0 if condensate exists, -1.0 otherwise.

# Arguments
- `θ_li`: Liquid-ice potential temperature [K]

# Returns
- `Float64`: Sign indicating presence (1.0) or absence (-1.0) of condensate
"""
function objective(θ_li, thermo_params, p, q_tot)
    ts = TD.PhaseEquil_pθq(thermo_params, p, θ_li, q_tot)
    return TD.has_condensate(thermo_params, ts) ? 1.0 : -1.0
end

"""
    find_condensation_thetali(thermo_params, p, q_tot)

Find the liquid-ice potential temperature at which condensation begins.

# Arguments
- `thermo_params`: Thermodynamic parameters
- `p`: Pressure [Pa]
- `q_tot`: Total water specific humidity [kg/kg]

# Returns
- `Float64`: Liquid-ice potential temperature at condensation point [K]
"""
function find_thetali_lengthscale(thermo_params, p, q_tot)
    θ_li_guess = 280.0  # Initial guess [K]
    θ_li_min = θ_li_guess - 100.0  # Lower bound [K]
    θ_li_max = θ_li_guess + 100.0  # Upper bound [K]
    
    θ_li_cond = find_zero(x -> objective(x, thermo_params, p, q_tot), (θ_li_min, θ_li_max), Bisection())
    return θ_li_cond
end

"""
    find_condensation_q(params, ds_profiles)

Calculate the difference between saturation and actual specific humidity.

# Arguments
- `params`: Model parameters
- `ds_profiles`: Dictionary containing mean temperature, specific volume, and total water profiles

# Returns
- `Array{Float64}`: Difference between saturation and actual specific humidity [kg/kg]
"""
function find_q_lengthscale(params, ds_profiles)
    sat_hus = TD.q_vap_saturation.(
        CAP.thermodynamics_params(params),
        ds_profiles["temperature_mean"][:, :],
        1 ./ ds_profiles["alpha_mean"][:, :],
        TD.PhaseEquil
    )
    
    hus = ds_profiles["qt_mean"][:, :]
    return sat_hus - hus
end

