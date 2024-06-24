"""
x⁽ⁱ⁾ = [x⁽ⁱ⁾, y⁽ⁱ⁾, θ⁽ⁱ⁾, v⁽ⁱ⁾]
u⁽ⁱ⁾ = [u_acceleration⁽ⁱ⁾, u_steer⁽ⁱ⁾]
"""


using BlockArrays
using LinearAlgebra
using GLMakie

include("Environment.jl")
include("Players.jl")


struct surveillance_demo{}
    env :: base_environment
    players :: Array{Player}
end

function active_surveillance_demo()

    demo = init(; initial_beliefs = nothing, initial_grid_size = (10, 10), L = 1)

    controls = nothing # TODO: get controls from players
    trajectory = unroll(demo.env, controls, 20)

    coords1 = [x[Block(1)][1:2] for x in trajectory]
    coords2 = [x[Block(2)][1:2] for x in trajectory]

    fig = Figure()
    ax = Axis(fig[1, 1], aspect = DataAspect(), xlabel = "x", ylabel = "y")
    lines!(ax, coords1, color = :blue)
    lines!(ax, coords2, color = :red)
    fig
end

function init(;initial_beliefs :: Array{BlockVector{Float64}} = nothing, initial_grid_size :: Tuple{Int}  = (10, 10),
    L :: Int = 1)

    grid_size :: Tuple{Int} = initial_grid_size

    βₒ¹ = BlockVector{Float64}([rand(1:grid_size[1]), rand(1:grid_size[2]), 0, 1,
        1.0, 1.0, 1.0, .5], [4, 4]) # 4 for state, 4 for covariance
    βₒ² = BlockVector{Float64}([rand(1:grid_size[1]), rand(1:grid_size[2]), 0, 1,
        1.0, 1.0, 1.0, .5], [4, 4])

    if !isnothing(initial_beliefs)
        βₒ¹ = initial_beliefs[1]
        βₒ² = initial_beliefs[2]
    end
    β = [βₒ¹, βₒ²]

    function state_dynamics(states :: BlockVector{Float64}, u :: BlockVector{Float64}; τ :: Float64 = 1.0, M :: Function = (u) -> 1.0 * norm(u), motion_noise :: Int = 1)
        new_state = BlockVector{Float64}(undef, [4 for _ in eachindex(blocks(states))])
        for i in eachindex(blocks(states))
            [x, y, θ, v] = states[Block(i)]
            [accel, steer] = u[Block(i)]

            # TODO: Find a good value for L
            ẋ = [ v * cos(θ), v * sin(θ), accel, v / (L * tan(steer))]'
            
            mₖ = Vector{Float64}([rand(-motion_nosie:motion_noise), rand(-motion_nosie:motion_noise)])

            # M scales motion noise mₖ according to size of u[i]
            new_state[Block(i)] .= states[Block(i)] + τ * ẋ + M(u[i]) * mₖ
        end

        return new_states
    end

    function measurement_noise_scaler(state :: Vector{Float64}; surveillance_radius :: Int = 10)
        # Only take x and y coords from state vector
        n = norm(state[1:2], 2) - surveillance_radius 
        n = max(1, n) # make sure noise multiplier doesn't get too small, don't want players to be able to see each other perfectly
        n = min(5, n) # make sure noise multiplier doesn't get too large

        return Matrix(n * I, 2, 2)
    end

    function observation_function(;states :: BlockVector{Float64}, N :: Function = measurement_noise_scaler)

        measurement_noise = BlockVector{Float64}(undef, length(states))
        for i in eachindex(blocks(measurement_noise))
            measurement_noise[Block(i)] .= N(states[Block(i)]) * rand(Float64, 2)
        end

        observations = BlockVector{Float64}(undef, [2 for _ in eachindex(blocks(states))])
        for i in eachindex(blocks(states))
            observations[Block(i)] .= states[Block(i)][1:2] + measurement_noise[Block(i)]
        end

        return observations
    end

    initial_state = BlockVector{Float64}(undef, [4 for _ in 1:2])
    initial_state[Block(1)] .= [-10.0, 20.0, 0.0, 1.0] # Player 1, surveiller
    initial_state[Block(2)] .= [-10.0, 15.0, 0.0, 1.0]


    env = init_base_environment(;
    state_dynamics = state_dynamics,
    observation_function = observation_function,
    num_agents = 2,
    state_dim = 4,
    initial_state = initial_state,
    final_time = -1)

    players = [init_player(;
    player_type = SDGiBS,
    player_id = i,
    belief = β[i],
    cost = () -> Inf, # TODO: cost
    action_space = 2,
    default_action = [0, 0].
    time = 20) for i in 1:2]
    
    return surveillance_demo{env, players}
end
