import Pkg; Pkg.activate(".")

using Flux, ProgressMeter, Statistics
using DataFrames
using Glob
using Flux: Dropout
using Random
using StatsBase
using CSV

include("mixing_length.jl")

predictors = [
    "z", 
    "t", 
    "p",
    "q_tot",
    "buoyancy_freq",
    "strain_rate", 
    "mix_len", 
    "mix_len_no_z",
    "thetali_gradient",
    "qt_gradient",
    "theta_lengthscale",
    "q_lengthscale",
    "gr1",
    "gr2",
    "gr3",
    "tke",
    "theta_li",
]

target = "cloud_fraction"

# get data from more sites
function get_les_site_names()
    pattern = "../../../../net/sampo/data1/zhaoyi/GCMForcedLES/cfsite/*/HadGEM2-A/amip/Output.cfsite*_HadGEM2-A_amip_2004-2008.*.4x/stats/Stats.cfsite*_HadGEM2-A_amip_2004-2008.*.nc"
    return Glob.glob(pattern)
end

site_names = get_les_site_names()

# t_ts = NCDataset(site_names[1]).group["timeseries"]
# t_prof = NCDataset(site_names[1]).group["profiles"]


# sample n sites at random to process, setting seed 
# Random.seed!(42)
# n_sites = 5
# sample_site_names = sample(site_names, n_sites, replace=false)

sample_site_data = []
@showprogress "Loading sites..." for site_name in site_names
    try
        push!(sample_site_data, process_les_file(site_name, params));
    catch e
        @warn "Failed to process site $site_name: $e"
    end
end;

println("Successfully processed $(length(sample_site_data)) sites")

# get unique sites 
# unique_months = unique([split(site, "/")[11] for site in site_names])
# unique_sites = unique([parse(Int, split(split(site, ".")[10], "_")[1][7:end]) for site in site_names])

# Combine data from all sites
X, y, sites, months, predictors_in_order = flatten_and_preprocess_multiple_sites(sample_site_data, target)

# Create DataFrame with all data
df = DataFrame(X, predictors_in_order)
df[!, :cloud_fraction] = y
df[!, :site] = sites
df[!, :month] = months

# save df 
CSV.write("all_sites_data.csv", df)

# load df 
df = CSV.read("all_sites_data.csv", DataFrame)

df2 = filter(row -> row.q_tot < 1e-5, df)
df2.cloud_fraction

println("Total data points: $(nrow(df))")
println("Number of sites: $(length(unique(df.site)))")

# Create train/test split (using site-based split to avoid data leakage)
Random.seed!(42)
unique_sites_list = unique(df.site)
n_train_sites = round(Int, 0.7 * length(unique_sites_list))
train_sites = shuffle(unique_sites_list)[1:n_train_sites]
test_sites = setdiff(unique_sites_list, train_sites)

train = findall(x -> x in train_sites, df.site)
test = findall(x -> x in test_sites, df.site)

println("Training sites: $(length(train_sites)), Test sites: $(length(test_sites))")
println("Training samples: $(length(train)), Test samples: $(length(test))")

n_features = length(predictors)
n_hidden = 64

# Create neural network model - removed sigmoid, using linear output
model = Chain(
    Dense(n_features, n_hidden, relu),
    BatchNorm(n_hidden),  # Add BatchNorm to help with training
    Dense(n_hidden, n_hidden, relu),
    BatchNorm(n_hidden),
    Dense(n_hidden, 1)    # Linear output layer, no activation
)

# Modify to use CUDA and cuDNN
# using CUDA
# CUDA.allowscalar(false) # prevent slow scalar operations on GPU

# 1. Prepare data for Flux
# Flux expects features as rows and samples as columns, and Float32 for speed
X_train = Float32.(Matrix(df[train, predictors])')
y_train = Float32.(reshape(df[train, :cloud_fraction], 1, :))

# Scale features to help with training
feature_means = mean(X_train, dims=2)
feature_stds = std(X_train, dims=2)
X_train_scaled = (X_train .- feature_means) ./ feature_stds

# 2. Move model to GPU and define optimizer
#c_model = cu(model)
opt_state = Flux.setup(Flux.Adam(0.001), model)

# 3. Create a data loader for batching and move data to GPU
# X_train_scaled_gpu = cu(X_train_scaled)
# y_train_gpu = cu(y_train)
loader = Flux.DataLoader((X_train_scaled, y_train), batchsize=64, shuffle=true)

# 4. Training loop with clamping - now using GPU model
losses = []
@showprogress "Training..." for epoch in 1:10
    batch_losses = []
    for (x_batch, y_batch) in loader
        loss, grads = Flux.withgradient(model) do m
            # Get raw predictions and clamp to [0,1]
            y_hat = clamp.(m(x_batch), 0.0f0, 1.0f0)
            Flux.mse(y_hat, y_batch)
        end
        Flux.update!(opt_state, model, grads[1])
        push!(batch_losses, loss)
    end
    push!(losses, mean(batch_losses))
    
    # Print progress every 10 epochs
    if epoch % 10 == 0
        println("Epoch $epoch: Loss = $(losses[end])")
    end
end

# save model
s = Flux.state(model)
JLD2.jldsave("5_sites_big_network_model.jld2", model_state = s)

# load model 
m2 = Chain(
    Dense(n_features, n_hidden, relu),
    BatchNorm(n_hidden),  # Add BatchNorm to help with training
    Dense(n_hidden, n_hidden, relu),
    BatchNorm(n_hidden),
    Dense(n_hidden, 1)    # Linear output layer, no activation
)

Flux.loadmodel!(m2, JLD2.load("5_sites_big_network_model.jld2", "model_state"))


# Evaluate on test data - using GPU
X_test = Float32.(Matrix(df[test, predictors])')
X_test_scaled = (X_test .- feature_means) ./ feature_stds
y_test = Float32.(reshape(df[test, :cloud_fraction], 1, :))

# Move test data to GPU
X_test_scaled_gpu = cu(X_test_scaled)
y_test_gpu = cu(y_test)

# Run prediction on GPU
y_pred_test = clamp.(m2(X_test_scaled), 0.0f0, 1.0f0)
test_mse = Flux.mse(y_pred_test, y_test)
println("\nTest MSE: $test_mse")

# Optional: save the trained model back to CPU if you need it later
cpu_model = cpu(c_model)





#### Visualize results
using CairoMakie

# Create figure
fig = Figure(size = (800, 600))

# Create axis with equal aspect ratio
ax = Axis(fig[1, 1], 
    xlabel = "True Values", 
    ylabel = "Predicted Values",
    title = "Model Predictions vs True Values",
    aspect = 1)

# Plot scatter points
scatter!(ax, vec(y_test), vec(y_pred_test), 
    markersize = 8,
    color = :blue,
    alpha = 0.5,
    label = "Data Points")

# Add perfect prediction line (1:1 line)
lines!(ax, [0, 1], [0, 1], 
    color = :red, 
    linestyle = :dash,
    label = "1:1 Line")

# Add legend
axislegend(ax, position = :lt)

# Add correlation coefficient as text
corr = cor(vec(y_test), vec(y_pred_test))
text!(ax, 0.05, 0.95, 
    text = "Correlation: $(round(corr, digits=3))",
    align = (:left, :top))

# Save figure
save("test2.png", fig)

