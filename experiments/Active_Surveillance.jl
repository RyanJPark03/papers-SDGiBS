"""
x⁽ⁱ⁾ = [x⁽ⁱ⁾, y⁽ⁱ⁾, θ⁽ⁱ⁾, v⁽ⁱ⁾]
u⁽ⁱ⁾ = [u_acceleration⁽ⁱ⁾, u_steer⁽ⁱ⁾]
"""


using BlockArrays
using LinearAlgebra
using Distributions
using GLMakie

include("Environment.jl")
include("Players.jl")


struct surveillance_demo{}
	env::base_environment
	players::Array{Player}
end


function active_surveillance_demo_main()
	open("./out.txt", "w") do file
		redirect_stdout(file) do 
			run_active_surveillance_demo()
		end
	end
end

function run_active_surveillance_demo()

	demo = init(; L = 1)
    trajectory = []
    push!(trajectory, demo.env.current_state)

    for tt in 1:demo.env.final_time - 1
		time_step_all(demo.players, demo.env)
        push!(trajectory, demo.env.current_state)
    end
	
	coords1 = [[x[1] for x in trajectory], [x[2] for x in trajectory]]
	coords2 = [[x[5] for x in trajectory], [x[6] for x in trajectory]]

	fig = Figure()
	ax = Axis(fig[1, 1], xlabel = "x", ylabel = "y")
	scatterlines!(coords1[1],coords1[2]; color = :blue)
	scatterlines!(coords2[1],coords2[2]; color = :red)
	return fig
end

function init(; L::Int = 1)
	function state_dynamics(states::Vector{T}, u, m;
		τ::Float64 = 1.0, M::Function = (u) -> 1.0 * norm(u)^2, L::Float64 = 1.0, block=true) where T
	new_state = Vector{T}(undef, length(states))
	for i in 1:2
		x, y, θ, v = states[4 * (i - 1) + 1 : 4 * i]
		accel, steer = u[2 * (i - 1) + 1 : 2 * i]

		ẋ = [v * cos(θ), v * sin(θ), v / (L * tan(steer)), accel]

		new_state[4 * (i - 1) + 1 : 4 * i] .= states[4 * (i - 1) + 1 : 4 * i] + τ * ẋ + M(u[i]) * m[4 * (i - 1) + 1 : 4 * i]
	end
	return new_state
end

	function state_dynamics(states::BlockVector{T}, u::BlockVector, m::BlockVector;
			τ::Float64 = 0.1, M::Function = (u) -> 1.0 * norm(u)^2, L::Float64 = 10.0, ϵₛ::Float64 = 1e-6,
			 block=true) where T
		new_state = Union{BlockVector, Vector}
		if block
			new_state = BlockVector{Any}(undef, [4 for _ in eachindex(blocks(states))])
		else
			new_state = Vector{T}(undef, 4 * length(blocks(states)))
		end
		for i in eachindex(blocks(states))
			x, y, θ, v = states[Block(i)]
			accel, steer = u[Block(i)]

			# println("steer: ", steer) #steer is going to infinity
			steer = max(5π/6, min(-5π/6, steer))
			δv = (abs(steer) < ϵₛ) ? 0.0 : v / (L * tan(steer))
			ẋ = [v * cos(θ), v * sin(θ), δv, accel]# assign 2 5 for drawing

			# M scales motion noise mₖ according to size of u[i], i.e. more noise the bigger the control
			if block
				new_state[Block(i)] .= states[Block(i)] + τ * ẋ + M(u[Block(i)]) * m[Block(i)]
				new_state[Block(i)][3] %= 2π
			else
				new_state[4 * (i - 1) + 1 : 4 * i] .= states[Block(i)] + τ * ẋ + M(u[Block(i)]) * m[Block(i)]
			end
		end
		return new_state
	end

	function measurement_noise_scaler(state::Vector; surveillance_radius::Int = 10)
		# Only take x and y coords from state vector
		n = norm(state[1:2] .- 5, 2) - surveillance_radius
		n = max(0.1, n) # make sure noise multiplier doesn't get too small, don't want players to be able to see each other perfectly
		n = min(100000, n) # make sure noise multiplier doesn't get too large

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

	function observation_function(; states::BlockVector{T}, m, N::Function = measurement_noise_scaler, block=true) where T
		observations = Union{BlockVector, Vector}
		if block
			observations = BlockVector{Float64}(undef, [4 for _ in eachindex(blocks(states))])
		else
			observations = Vector{T}(undef, 4 * length(blocks(states)))
		end
		for i in eachindex(blocks(states))
			if block
				observations[Block(i)] .= states[Block(i)] + N(states[Block(i)]) * m[Block(i)]
			else
				observations[4 * (i - 1) + 1 : 4 * i] .= states[Block(i)] + N(states[Block(i)]) * m[Block(i)]
			end
		end

		return observations
	end

	initial_state = BlockVector{Float64}(undef, [4 for _ in 1:2])
	initial_state[Block(1)] .= [10.0, 20.0, 0.0, 1.0] # Player 1, surveiller
	initial_state[Block(2)] .= [10.0, 15.0, 0.0, 1.0]

	env = init_base_environment(;
			state_dynamics = state_dynamics,
			observation_function = observation_function,
			num_agents = 2,
			state_dim = 4,
		    dynamics_noise_dim = 4,
			observation_noise_dim = 4,
			initial_state = initial_state,
			final_time = 16) # 15 if inital action is 0 0.5

	initial_beliefs = BlockVector{Float64}(undef, [20, 20])
	initial_cov_matrix = [
		1.0 0.0 0.0 0.0;
		0.0 1.0 0.0 0.0;
		0.0 0.0 .06 0.0;
		0.0 0.0 0.0 0.5;
	]
	initial_beliefs[Block(1)] .= vcat(copy(initial_state[Block(1)]), vec(copy(initial_cov_matrix)))
	initial_beliefs[Block(2)] .= vcat(copy(initial_state[Block(2)]), vec(copy(initial_cov_matrix)))

	function cₖ¹(β, u)
		R = Matrix(0.1 * I, 2, 2)
		if typeof(u) == BlockVector
			return u[Block(1)]' * R * u[Block(1)]
		else
			return u[1:2]' * R * u[1:2]
		end
	end
	function cₗ¹(β)
		if typeof(β) == BlockVector
			return prod(diag(reshape(β[Block(2)][5:end], (4, 4))))
		else
			return prod(diag(reshape(β[Int(length(β)//2 + 5):end], (4, 4))))
		end
	end

	α₁ = 10.0
	α₂ = 10.0
	vₖ_des = 1.0
	function c_coll(β)
		# "c_coll = exp(-d(xₖ)). Here d(xₖ) is the expcted euclidean distance
		# until collision between the two agents, taking their outline into account."
		# TODO wtf does "taking their outline into account" mean???
		if typeof(β) == BlockVector
			return norm(β[Block(1)][1:2] - β[Block(2)][1:2], 2)
		else
			return norm(β[1:2] - β[Int(length(β)//2 + 1):Int(length(β)//2 + 2)], 2)
		end
	end
	function cₖ²(β::T, u::T) where T
		R = Matrix(0.1 * I, 2, 2)
		if typeof(β) == BlockVector
			return u[Block(2)]' * R * u[Block(2)] + α₁ * (β[Block(2)][4] - vₖ_des)^2 + α₂ * c_coll(β)
		else 
			return u[3:end]' * R * u[3:end] + α₁ * (β[Int(length(β)//2 + 4)] - vₖ_des)^2 + α₂ * c_coll(β)
		end
	end

	function cₗ²(β)
		if typeof(β) == BlockVector
			return α₁ * norm(β[Block(2)][4] - vₖ_des, 2)^2 + α₂ * c_coll(β)
		else
			return α₁ * norm(β[Int(length(β)//2 + 4)] - vₖ_des, 2)^2 + α₂ * c_coll(β)
		end
	end
	costs = [cₖ¹, cₖ²]
	final_costs = [cₗ¹, cₗ²]

	players = [init_player(;
		player_type = type_SDGiBS,
		player_id = i,
		belief = initial_beliefs[Block(i)],
		cost = costs[i],
		final_cost = final_costs[i],
		action_space = 2,
		observation_space = 4,
		default_action = [0.0, -0.05],# accel, steer
		time = env.final_time,
		num_players = 2) for i in 1:2]

	return surveillance_demo(env, players)
end
