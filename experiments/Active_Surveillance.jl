"""
x⁽ⁱ⁾ = [x⁽ⁱ⁾, y⁽ⁱ⁾, θ⁽ⁱ⁾, v⁽ⁱ⁾]
u⁽ⁱ⁾ = [u_acceleration⁽ⁱ⁾, u_steer⁽ⁱ⁾]
"""


using BlockArrays
using LinearAlgebra
# using GLMakie

include("Environment.jl")
include("Players.jl")


struct surveillance_demo{}
	env::base_environment
	players::Array{player}
end

function active_surveillance_demo()

	demo = init(; L = 1)
    trajectory = []
    push!(trajectory, demo.env.current_state)
	motion_noise = 1.0

    for tt in 1:demo.env.final_time
        # time_step could return a block vector already, not entirely sure
		m = BlockVector(vcat([rand(-motion_noise:motion_noise, demo.env.observation_noise_dim) for _ in 1:demo.env.num_agents]...),
						[demo.env.observation_noise_dim for _ in 1:demo.env.num_agents])
		observations = demo.env.observation_function(;states=demo.env.current_state,
				m = m)
        # controls = BlockVector(time_step_all(demo.players, demo.env, observations(demo.env)))
		controls = BlockVector(time_step_all(demo.players, demo.env, observations))
        push!(trajectory, unroll(demo.env, controls, 1;
        # TODO: noise should be normally distributed
            noise=Vector{Float64}([rand(-motion_noise:motion_noise), rand(-motion_noise:motion_noise)]))...)
    end
	
	coords1 = vcat([x[Block(1)][1:2] for x in trajectory]...)
	coords2 = vcat([x[Block(2)][1:2] for x in trajectory]...)

    # TODO: something to fix here rip
	# fig = Figure()
	# ax = Axis(fig[1, 1], xlabel = "x", ylabel = "y")
	# lines!(ax, coords1, color = :blue)
	# lines!(ax, coords2, color = :red)
	# fig
end

function init(; L::Int = 1)

	function state_dynamics(states::BlockVector{Float64}, u::BlockVector{Float64}, m::BlockVector{Float64};
			τ::Float64 = 1.0, M::Function = (u) -> 1.0 * norm(u)^2, L::Float64 = 1.0)
		new_state = BlockVector{Float64}(undef, [4 for _ in eachindex(blocks(states))])
		for i in eachindex(blocks(states))
			x, y, θ, v = states[Block(i)]
			accel, steer = u[Block(i)]

			# TODO: Find a good value for L
			# ẋ = [v * cos(θ), v * sin(θ), accel, v / (L * tan(steer))]
			ẋ = [v * cos(θ), v * sin(θ), v / (L * tan(steer)), accel] # assign 4 for Derivative# assign 2 5 for drawing

			# M scales motion noise mₖ according to size of u[i], i.e. more noise the bigger the control
			new_state[Block(i)] .= states[Block(i)] + τ * ẋ + M(u[i]) * m[Block(i)]
		end
		return new_state
	end

	function measurement_noise_scaler(state::Vector{Float64}; surveillance_radius::Int = 10)
		# Only take x and y coords from state vector
		n = norm(state[1:2], 2) - surveillance_radius
		n = max(1, n) # make sure noise multiplier doesn't get too small, don't want players to be able to see each other perfectly
		n = min(5, n) # make sure noise multiplier doesn't get too large

        v = .25 * state[4]^2 # velocity scaled noise
        t = .01 * 360 # noise for theta is 1% of a circle

        noise = 
			[
			n 0 0 0; 
			0 n 0 0; 
			0 0 v 0; 
			0 0 0 t
			]

		return noise
	end

	function observation_function(; states::BlockVector{Float64}, m::BlockVector{Float64}, N::Function = measurement_noise_scaler)
		observations = BlockVector{Float64}(undef, [4 for _ in eachindex(blocks(states))])

		for i in eachindex(blocks(states))
			observations[Block(i)] .= states[Block(i)] + N(states[Block(i)]) * m[Block(i)]
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
		    dynamics_noise_dim = 4,
			observation_noise_dim = 4,
			initial_state = initial_state,
			final_time = 10)

	# cov matrix 2 0 ; 0 2
	initial_beliefs = BlockVector{Float64}(undef, [20, 20])
	initial_cov_matrix = [
		2.0 0.0 0.0 0.0;
		0.0 2.0 0.0 0.0;
		0.0 0.0 .06 0.0;
		0.0 0.0 0.0 1.0;
	]
	initial_beliefs[Block(1)] .= vcat(initial_state[Block(1)], vec(initial_cov_matrix))
	initial_beliefs[Block(2)] .= vcat(initial_state[Block(2)], vec(initial_cov_matrix))

	function cₖ¹(β::BlockVector{Float64}, u::BlockVector{Float64})
		R = Matrix(0.1 * I, 2, 2)
		return u[Block(1)]' * R * u[Block(1)]
	end
	function cₗ¹(β::BlockVector{Float64})  
		# return det(reshape(β[Block(2)][5:8], (2, 2)))
		return prod(diag(reshape(β[Block(2)][5:8], (2, 2))))
	end

	α₁ = 1.0
	α₂ = 1.0
	vₖ_des = 1.0
	function c_coll(β::BlockVector{Float64})
		# "c_coll = exp(-d(xₖ)). Here d(xₖ) is the expcted euclidean distance
		# until collision between the two agents, taking their outline into account."
		# TODO wtf does "taking their outline into account" mean???
		return norm(β[Block(1)][1:2] - β[Block(2)][1:2], 2)
	end
	function cₖ²(β::BlockVector{Float64}, u::BlockVector{Float64})
		R = Matrix(0.1 * I, 2, 2)
		return u[Block(2)]' * R * u[Block(2)] + α₁(β[Block(2)][4] - vₖ_des)^2 + α₂ * c_coll(β)
	end

	function cₗ²(β::BlockVector{Float64})
		return α₁ * (β[Block(2)][4] - vₖ_des)^2 + α₂ * c_coll(β)
	end
	costs = [cₖ¹, cₖ²]
	final_costs = [cₗ¹, cₗ²]

	players = [init_player(;
		player_type = type_SDGiBS,
		player_id = i,
		# Block of a Block vector is a vector
		belief = initial_beliefs[Block(i)],
		cost = costs[i],
		final_cost = final_costs[i],
		action_space = 2,
		observation_space = 4,
		default_action = [0.0, 0.5],# accel, steer
		time = 20) for i in 1:2]

	return surveillance_demo(env, players)
end
