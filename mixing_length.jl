using Glob
using NCDatasets
using Statistics
using CairoMakie
using ProgressMeter

import ClimaAtmos as CA 
import ClimaAtmos.Parameters as CAP
import Thermodynamics as TD

using Roots # root solver for theta_li computation

# ML Packages
using DataFrames
using MLJ
using MLJDecisionTreeInterface
using GLM
using Flux


# Path to LES output data
# LES_DATA_PATH = "/net/sampo/data1/zhaoyi/GCMForcedLES/cfsite/07/HadGEM2-A/amip/Output.cfsite17_HadGEM2-A_amip_2004-2008.07.4x/stats/Stats.cfsite17_HadGEM2-A_amip_2004-2008.07.nc"

# Load LES datasets
# ds_profiles = NCDataset(LES_DATA_PATH).group["profiles"]
# ds_timeseries = NCDataset(LES_DATA_PATH).group["timeseries"]

# Initialize ClimaAtmos configuration and parameters
config = CA.AtmosConfig("ClimaAtmos.jl/config/model_configs/prognostic_edmfx_bomex_pigroup_column.yml")
params = CA.ClimaAtmosParameters(config)
# CAP.turbconv_params(params)

"""
    tke(ds)

Compute turbulent kinetic energy from velocity variances.
"""
function tke(ds)
    return (ds["w_mean2"][:, :] .+ ds["v_mean2"][:, :] .+ ds["u_mean2"][:, :]) ./ 2
end

"""
    add_z_dim(ds, varname, len)

Extend a time series variable to all heights by repeating.
"""
function add_z_dim(ds, varname, len)
    repeat(ds[varname][:]', len, 1)
end

"""
    add_t_dim(ds, varname, len)

Extend a profile variable to all times by repeating.
"""
function add_t_dim(ds, varname, len)
    repeat(ds[varname][:], 1, len)
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
    ustar = add_z_dim(ds_timeseries, "friction_velocity_mean", ds_profiles.dim["z"])
    
    # Get height coordinate and extend to all times
    ᶜz = add_t_dim(ds_profiles, "z", ds_profiles.dim["t"])
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
    obukhov_length = add_z_dim(ds_timeseries, "obukhov_length_mean", ds_profiles.dim["z"])
    
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
    θ_li_min = θ_li_guess - 250.0  # Lower bound [K]
    θ_li_max = θ_li_guess + 300.0  # Upper bound [K]
    try
        θ_li_cond = find_zero(x -> objective(x, thermo_params, p, q_tot), (θ_li_min, θ_li_max), Bisection())
        return θ_li_cond
    catch
        # basically we expect water vapor to be zero, so return large length scale (e.g., we are far from saturation)
        return 1000
    end
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

"""
    process_les_file(filepath, params)

Process a single LES file to compute all relevant terms for ML training.

# Arguments
- `filepath`: Path to the LES netCDF file
- `params`: ClimaAtmos parameters

# Returns
- Dictionary containing all computed terms and metadata
"""
function process_les_file(filepath, params)
    # Load datasets
    ds = NCDataset(filepath)
    ds_profiles = ds.group["profiles"]
    ds_timeseries = ds.group["timeseries"]
    
    # Get basic coordinates
    z = ds_profiles["z"][:]
    #time_dims = ds_profiles["t"][:]

    # get thermo params 
    thermo_params = CAP.thermodynamics_params(params)

    # unpack variables
    t = ds_profiles["temperature_mean"][:, :]
    buoyancy_freq = ds_profiles["buoyancy_frequency_mean2"][:, :]
    strain_rate = ds_profiles["strain_rate_magnitude"][:, :]
    q_tot = ds_profiles["qt_mean"][:, :]
    thetali = ds_profiles["thetali_mean"][:, :]
    cloud_fraction = ds_profiles["cloud_fraction"][:, :]
    alpha = ds_profiles["alpha_mean"][:, :]

    # approximate the pressure 
    p = CAP.R_d(params) .* 1 ./ alpha .* t

    
    # compute the mixing length
    mix_len, mix_len_no_z = compute_mixing_length(ds_profiles, ds_timeseries, params)
    
    # compute vertical gradients of thetali and q_tot
    thetali_gradient = compute_vertical_gradient(thetali', z)'
    qt_gradient = compute_vertical_gradient(q_tot', z)'
    
    # compute tke
    tke_data = tke(ds_profiles)
    
    # Compute length scales
    θ_lengthscale = find_thetali_lengthscale.(thermo_params, p, q_tot) .- thetali
    θ_lengthscale = clamp.(θ_lengthscale, -50.0, 50.0)
    q_lengthscale = find_q_lengthscale(params, ds_profiles)
    
    # Compute pi groups
    gr1 = qt_gradient .* mix_len_no_z ./ q_lengthscale
    gr2 = thetali_gradient .* mix_len_no_z ./ θ_lengthscale
    gr3 = qt_gradient .* thetali_gradient .* mix_len_no_z .^2 ./ (q_lengthscale .* θ_lengthscale)
    
    # Cloud fraction target
    cloud_fraction = ds_profiles["cloud_fraction"][:, :]
    
    cfsite, month = get_cfsite_and_month(filepath)
    
    # Create dictionary with all terms
    return Dict(
        "z" => add_t_dim(ds_profiles, "z", ds_profiles.dim["t"]),
        "t" => t,
        "p" => p,
        "q_tot" => q_tot,
        "buoyancy_freq" => buoyancy_freq,
        "strain_rate" => strain_rate,
        "mix_len" => mix_len,
        "mix_len_no_z" => mix_len_no_z,
        "thetali_gradient" => thetali_gradient,
        "qt_gradient" => qt_gradient,
        "theta_lengthscale" => θ_lengthscale,
        "q_lengthscale" => q_lengthscale,
        "gr1" => gr1,
        "gr2" => gr2,
        "gr3" => gr3,
        "cloud_fraction" => cloud_fraction,
        "tke" => tke_data,
        "buoyancy_freq" => buoyancy_freq,
        "theta_li" => thetali,
        "site" => basename(filepath),
        "cfsite" => cfsite,
        "month" => month,
    )
end

function get_cfsite_and_month(filepath)
    cfsite = parse(Int, split(split(basename(filepath), ".")[2], "_")[1][7:end])
    month = parse(Int, split(basename(filepath), ".")[3])
    return cfsite, month
end

"""
    flatten_and_preprocess(les_data_dicts, predictors, target)

Flatten and preprocess LES data for machine learning.

# Arguments
- `les_data_dicts`: Dictionary of LES data from process_les_file
- `predictors`: Array of symbols for predictor variables
- `target`: Symbol for target variable

# Returns
- Tuple containing:
  - X: DataFrame of predictors in MLJ format
  - y: Target variable as continuous
  - df: Full DataFrame with metadata
"""
function flatten_and_preprocess(les_data_dicts, predictors, target)

    X = []
    for predictor in predictors
        push!(X, vec(les_data_dicts[predictor]))
    end
    X = hcat(X...)
    y = vec(les_data_dicts[target])

    # Ensure cloud fraction is between 0 and 1
    if target == "cloud_fraction"
        y = clamp.(y, 0.0, 1.0)
    end

    return X, y
end

"""
    preprocess_for_ml(les_data_dicts; 
                     predictors=[:tke, :thetali_gradient, :gr3, :q_lengthscale, 
                               :buoyancy_freq, :gr2, :mix_len_no_z, :strain_rate,
                               :theta_lengthscale, :gr1, :qt_gradient, :q_tot, :t],
                     target=:cloud_fraction,
                     min_cloud_fraction=0.01,
                     max_height=3000.0)

Preprocess LES data for machine learning using MLJ.jl.

# Arguments
- `les_data_dicts`: Dictionary of LES data from process_les_file
- `predictors`: Array of symbols for predictor variables
- `target`: Symbol for target variable

# Returns
- Tuple containing:
  - X: DataFrame of predictors in MLJ format
  - y: Target variable as continuous
  - df: Full DataFrame with metadata
"""
function preprocess_for_ml(les_data_dicts; 
                         predictors=[:tke, :thetali_gradient, :gr3, :q_lengthscale, 
                                   :buoyancy_freq, :gr2, :mix_len_no_z, :strain_rate,
                                   :theta_lengthscale, :gr1, :qt_gradient, :q_tot, :t],
                         target=:cloud_fraction,
                         min_cloud_fraction=0.01,
                         max_height=3000.0)
    
    # Get height mask
    height_mask = les_data_dicts["z"] .<= max_height
    
    # Get cloud fraction mask
    cloud_mask = les_data_dicts[string(target)] .>= min_cloud_fraction
    
    # Combine masks
    valid_mask = height_mask .& cloud_mask
    
    # Initialize DataFrame
    df = DataFrame()
    
    # Add predictors
    for pred in predictors
        df[!, string(pred)] = vec(les_data_dicts[string(pred)][valid_mask])
    end
    
    # Add target
    df[!, string(target)] = vec(les_data_dicts[string(target)][valid_mask])
    
    # Add metadata columns
    df[!, :height] = repeat(les_data_dicts["z"][height_mask], outer=size(les_data_dicts[string(target)], 1))
    df[!, :time] = repeat(les_data_dicts["t"], inner=sum(height_mask))
    df[!, :site] = fill(les_data_dicts["site"], size(df, 1))
    
    # Ensure cloud fraction is between 0 and 1
    df[!, string(target)] = clamp.(df[!, string(target)], 0.0, 1.0)
    
    # Convert to MLJ compatible format
    X = select(df, Not(string(target)))
    y = df[!, string(target)]
    
    # Coerce types for MLJ
    X = coerce(X, autotype(X))
    y = coerce(y, Continuous)
    
    return X, y, df
end

# Example usage for MLJ:
# using MLJ
# X, y, df = preprocess_for_ml(les_data)
# train, test = partition(eachindex(y), 0.7, shuffle=true)
# model = @load RandomForestRegressor pkg=DecisionTree
# mach = machine(model, X, y)
# fit!(mach, rows=train)
# y_pred = predict(mach, X[test, :])

"""
    flatten_and_preprocess_multiple_sites(les_data_list, predictors, target)

Flatten and preprocess LES data from multiple sites for machine learning.

# Arguments
- `les_data_list`: Array of LES data dictionaries from process_les_file
- `predictors`: Array of predictor variable names (strings)
- `target`: Target variable name (string)

# Returns
- Tuple containing:
  - X: Matrix of predictors
  - y: Target variable vector
  - sites: Vector of site names corresponding to each data point
"""
function flatten_and_preprocess_multiple_sites(les_data_list, target)
    all_X = []
    all_y = []
    all_site = []
    all_month = []
    predictor_order = collect(keys(les_data_list[1]))
    # remove cfsite, month, site, cloud_fraction from predictor_order
    predictor_order = [p for p in predictor_order if p ∉ ["cfsite", "month", "site", "cloud_fraction"]]
    for les_data in les_data_list
        # Process each site's data
        X_site = []
        for predictor in predictor_order
            if predictor in ["cfsite", "month", "site", "cloud_fraction"]
                continue # not predictors - we'll handle separately
            end
            push!(X_site, vec(les_data[predictor]))
        end
        X_site = hcat(X_site...)
        y_site = vec(les_data[target])
        
        # Ensure cloud fraction is between 0 and 1
        if target == "cloud_fraction"
            y_site = clamp.(y_site, 0.0, 1.0)
        end

        # add site month and cfsite number, repeating
        site = fill(les_data["cfsite"], size(X_site, 1))
        month = fill(les_data["month"], size(X_site, 1))
        
        # Collect data from this site
        push!(all_X, X_site)
        push!(all_y, y_site)
        push!(all_site, site)
        push!(all_month, month)
    end
    
    # Combine all sites
    X_combined = vcat(all_X...)
    y_combined = vcat(all_y...)
    sites_combined = vcat(all_site...)
    months_combined = vcat(all_month...)
    
    return X_combined, y_combined, sites_combined, months_combined, predictor_order
end








