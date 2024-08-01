"""
x⁽ⁱ⁾ = [x⁽ⁱ⁾, y⁽ⁱ⁾, θ⁽ⁱ⁾, v⁽ⁱ⁾]
u⁽ⁱ⁾ = [u_acceleration⁽ⁱ⁾, u_steer⁽ⁱ⁾]
hrs: 8
"""


using BlockArrays
using LinearAlgebra
using Distributions
using GLMakie
using ProgressBars
using JuMP
using OSQP: OSQP

include("Environment.jl")
include("Players.jl")


struct surveillance_demo{}
	env::base_environment
	players::Array{Player}
end


function active_surveillance_demo_main()
	open("./out.temp", "w") do file
		redirect_stdout(file) do 
			run_active_surveillance_demo(30, .3)
		end
	end
end

function run_active_surveillance_demo(time_steps, τ)
	surveillance_center = [5, 5]
	surveillance_radius = 10
	
	demo = init(time_steps, τ; surveillance_center = surveillance_center, surveillance_radius = surveillance_radius, L = 1)
    trajectory = []
    push!(trajectory, demo.env.current_state)

    for tt in ProgressBar(1:demo.env.final_time - 1)
		time_step_all_coop(demo.players, demo.env)
        push!(trajectory, demo.env.current_state)
    end
	coords1 = [[x[1] for x in trajectory], [x[2] for x in trajectory]]
	coords2 = [[x[5] for x in trajectory], [x[6] for x in trajectory]]

	belief_coords1 = [[x[3][1] for x in demo.players[1].history], [x[3][2] for x in demo.players[1].history]]
	belief_coords2 = [[x[3][1] for x in demo.players[2].history], [x[3][2] for x in demo.players[2].history]]

	observations1 = [[demo.players[1].history[x][2][1] for x in 2:length(demo.players[1].history)],
	[demo.players[1].history[x][2][2] for x in 2:length(demo.players[1].history)]]
	observations2 = [[demo.players[2].history[x][2][1] for x in 2:length(demo.players[1].history)],
	[demo.players[2].history[x][2][2] for x in 2:length(demo.players[1].history)]]

	costs = []
	for tt in 1 : demo.env.final_time
		println("calculating cost for time: ", tt)
		time_slices = [get_history(player, tt) for player in demo.players]

		beliefs = vcat([time_slice[3] for time_slice in time_slices]...)
		action = vcat([isnothing(time_slices[ii][1]) ? zeros(demo.players[ii].action_space) : time_slices[ii][1] for ii in eachindex(time_slices)]...)


		p2_cov = reshape(beliefs[Int(length(beliefs)//2 + 5):end], (4, 4))
		println("p2 cov: ")
		show(stdout, "text/plain", p2_cov)
		println()

		p1_costs = (;time_dependent_cost = demo.players[1].cost(beliefs, action), final_cost = demo.players[1].final_cost(beliefs))
		p2_costs = (;time_dependent_cost = demo.players[2].cost(beliefs, action), final_cost = demo.players[2].final_cost(beliefs))
		push!(costs, (p1_costs, p2_costs))
	end
	show(stdout, "text/plain", costs)


	fig = Figure()
	ax = Axis(fig[1, 1], xlabel = "x", ylabel = "y")
	sg = SliderGrid(
        fig[2, 1],
        (label = "time", range = 1:demo.env.final_time, format = x-> "", startvalue = 1)
    )
    observable_time = lift(sg.sliders[1].value) do a
        Int(a)
    end
	player_locations = lift(observable_time) do a
		time_slices = [get_history(player, a) for player in demo.players]
		# action = [slice[2] for slice in time_slices]
		# println("p2 actions: ")
		# display(action[2])
		beliefs = [time_slice[3] for time_slice in time_slices]

		positions = [belief[1:2] for belief in beliefs]
		covs = [reshape(belief[5:end], (4, 4)) for belief in beliefs]
		radii = [(cov[1, 1], cov[2, 2]) for cov in covs]

		return positions, radii
	end

	player_1_point = @lift Point2f($(player_locations)[1][1][1], $(player_locations)[1][1][2])
	player_2_point = @lift Point2f($(player_locations)[1][2][1], $(player_locations)[1][2][2])
	player_1_cov = @lift getellipsepoints($(player_locations)[1][1][1], $(player_locations)[1][1][2],
					$(player_locations)[2][1][1], $(player_locations)[2][1][2], 0)
	player_2_cov = @lift getellipsepoints($(player_locations)[1][2][1], $(player_locations)[1][2][2],
					$(player_locations)[2][2][1], $(player_locations)[2][2][2], 0)
	scatter!(ax, player_1_point; color = :blue)
	scatter!(ax, player_2_point; color = :red)
	lines!(ax, player_1_cov; color = (:blue, .75))
	lines!(ax, player_2_cov; color = (:red, .75))
	arc!(ax, Point2f(surveillance_center[1], surveillance_center[2]), surveillance_radius, 0, 2π; color = :black)


	# Static image
	bx = Axis(fig[1, 2], xlabel = "x", ylabel = "y")
	scatterlines!(bx, coords1[1], coords1[2]; color = :blue)
	scatterlines!(bx, belief_coords1[1], belief_coords1[2]; color = (:blue, 0.75), linestyle = :dash)
	# scatter!(belief_coords1[1], belief_coords1[2]; color = :black, markersize = )
	# scatterlines!(observations1[1], observations1[2]; color = (:blue, 0.5), linestyle = :dot)
	scatterlines!(bx, coords2[1], coords2[2]; color = :red)
	scatterlines!(bx, belief_coords2[1], belief_coords2[2]; color = (:red, 0.75), linestyle = :dash)
	arc!(bx, Point2f(surveillance_center[1], surveillance_center[2]), surveillance_radius, 0, 2π; color = :black)

	#for sizing:
	scatter!(ax, belief_coords1[1][end], belief_coords1[2][end]; color = :black, markersize = 1)
	scatter!(ax, belief_coords2[1][end], belief_coords2[2][end]; color = :black, markersize = 1)
	# display(fig, fullsreen = true)
	return fig
end

function init(time_steps, τₒ; surveillance_center = [0, 0], surveillance_radius::Int = 10, 
	L::Int = 1)
	# Magic Numbers
	p1_effort = .1
	p1_end_cost_weight = 1.0
	α₁ = 0.001
	α₂ = 30.0
	p2_effort = 1.0
	vₖ_des = 10.0
	collision_exponent_multiplier = 0.2

	initial_state = BlockVector{Float64}(undef, [4 for _ in 1:2])
	initial_state[Block(1)] .= [surveillance_center[1] - 8.0, surveillance_center[2] + 18.0, -0.01, 10.0] # Player 1, surveiller
	initial_state[Block(2)] .= [surveillance_center[1] - 10.0, surveillance_center[2] + 15.0, -0.01, 10.0]

	initial_beliefs = BlockVector{Float64}(undef, [20, 20])
	initial_cov_matrix = [
		10.0 0.0 0.0 0.0;
		0.0 10.0 0.0 0.0;
		0.0 0.0 0.006 0.0;
		0.0 0.0 0.0 0.5;
	]
	initial_beliefs[Block(1)] .= vcat(copy(initial_state[Block(1)]), vec(copy(initial_cov_matrix)))
	initial_beliefs[Block(2)] .= vcat(copy(initial_state[Block(2)]), vec(copy(initial_cov_matrix)))


	state_dynamics_noise_scaler = (u) ->  (norm(u)^2)^.25 .* I
	function state_dynamics(states::Vector{T}, u, m;
		τ::Float64 = τₒ, M::Function = state_dynamics_noise_scaler, L::Float64 = 1.0, block=true) where T
		new_state = Vector{T}(undef, length(states))
		for i in 1:2
			x, y, θ, v = states[4 * (i - 1) + 1 : 4 * i]
			accel, steer = u[2 * (i - 1) + 1 : 2 * i]

			dv = ( (v * tan(steer)) / L )
			ẋ = [v * cos(θ), v * sin(θ), dv, accel]

			new_state[4 * (i - 1) + 1 : 4 * i] .= states[4 * (i - 1) + 1 : 4 * i] + τ * ẋ + M(u[i]) * m[4 * (i - 1) + 1 : 4 * i]
		end
		return new_state
	end

	function state_dynamics(states::BlockVector{T}, u::BlockVector, m::BlockVector;
			τ::Float64 = τₒ, M::Function = state_dynamics_noise_scaler, L::Float64 = 10.0, ϵₛ::Float64 = 1e-6,
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

			δv = v * tan(steer) / L 
			ẋ = [v * cos(θ), v * sin(θ), δv, accel]# assign 2 5 for drawing

			# M scales motion noise mₖ according to size of u[i], i.e. more noise the bigger the control
			if block
				# new_state[Block(i)] .= states[Block(i)] + τ * ẋ + 0.0 .* m[Block(i)]
				new_state[Block(i)] .= states[Block(i)] + τ * ẋ + M(u[Block(i)]) * m[Block(i)]
			else
				new_state[4 * (i - 1) + 1 : 4 * i] .= states[Block(i)] + τ * ẋ + M(u[Block(i)]) * m[Block(i)]
			end
		end
		return new_state
	end

	function measurement_noise_scaler1(state::Vector; surveillance_center = [0, 0], surveillance_radius::Int = 10)
		# return I
		# Only take x and y coords from state vector
		n_outer = 100 * abs(norm(state[1:2] - surveillance_center, 2) - surveillance_radius^2)
		# n_outer = max(0.01, n_outer) # make sure noise multiplier doesn't get too small, don't want players to be able to see each other perfectly
		# n_outer = min(100, n_outer) # make sure noise multiplier doesn't get too large
		n = n_outer

        v = abs(1e-3 * state[4]) # velocity scaled noise
		t = abs(1e-3 * state[3])

        noise = 
			[
			n 0 0 0; 
			0 n 0 0; 
			0 0 t 0; 
			0 0 0 v
			]

		return noise
	end

	function measurement_noise_scaler2(state::Vector; surveillance_band_height = 0)
		n = .5 * (state[2] - surveillance_band_height)^2 + 1e-10 * state[1]
		v = 1e-10 * state[4]^2 # velocity scaled noise
        t = 1e-10 * state[3]
		noise = 
			[
			n 0 0 0; 
			0 n 0 0; 
			0 0 v 0; 
			0 0 0 t
			]
		return noise
	end

	function observation_function(; states::BlockVector{T}, m, N::Function = measurement_noise_scaler1, block=true) where T
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

	function cₖ¹(β, u)
		# return 0.0
		R = Matrix(p1_effort * I, 2, 2)
		if typeof(u) == BlockVector
			return u[Block(1)]' * R * u[Block(1)]
		else
			return u[1:2]' * R * u[1:2]
		end
	end
	
	function cₗ¹(β)
		# return 0.0
		if typeof(β) == BlockVector
			return p1_end_cost_weight * prod(diag(reshape(β[Block(2)][5:end], (4, 4)))[1:2])
		else
			return p1_end_cost_weight * prod(diag(reshape(β[Int(length(β)//2 + 5):end], (4, 4)))[1:2])
		end
	end

	function c_coll(β)
		half_length = Int(length(β)//2)
		ellipse1 = (β[1:2], reshape(β[5:half_length], (4, 4))[1:2, 1:2])
		ellipse2 = (β[half_length + 1: half_length + 2], reshape(β[half_length + 5: end], (4, 4))[1:2, 1:2])
		if overlap(ellipse1, ellipse2)
			distance = β[1:2] - β[half_length + 1:half_length + 2]
			return exp(-1.0 * collision_exponent_multiplier * norm(distance, 2))
		else
			return 0
		end
	end

	function overlap(ellipse1, ellipse2)
		# Gilitschenski, Igor, and Uwe D. Hanebeck. "A robust computational test for overlap of two arbitrary-dimensional
		# ellipsoids in fault-detection of kalman filters." 2012 15th International Conference on Information Fusion. IEEE, 2012.
		c = ellipse1[1]
		d = ellipse2[1]
		# Main.@infiltrate
		A = ellipse1[2]
		B = ellipse2[2]

		expected_distance = norm(c - d)

		radius_estimate1 = (A[1, 1] + A[2, 2]) / 2.0
		radius_estimate2 = (B[1, 1] + B[2, 2]) / 2.0
		min_safety_distance = radius_estimate1 + radius_estimate2

		return expected_distance < min_safety_distance


		# model = JuMP.Model()
		# JuMP.set_optimizer(model, OSQP.Optimizer)
		# JuMP.set_silent(model)
		# ϵ = 1e-5
		# @variable(model, ϵ <= s <= 1-ϵ) 
		# J = 1 .- (d - c)' * (((1/(1-s)) * (B \ I) + (1/s) * (A \ I)) \ I) * (d - c)
		# @objective(
		# 	model,
		# 	Min,
		# 	J[1]
    	# )
		
		# # @constraint(model, s >= ϵ, s <= 1-ϵ)
		# JuMP.optimize!(model)
		# (JuMP.termination_status(model) == JuMP.MOI.OPTIMAL) ||
        # 	error("OSQP did not find an optimal solution to this trajectory generation problem.")

		#  return JuMP.value(s) < 0
	end

	function cₖ²(β::T, u::T) where T
		# return 0.0
		R = Matrix(p2_effort * I, 2, 2)
		if typeof(β) == BlockVector
			return u[Block(2)]' * R * u[Block(2)] + α₁ * (β[Block(2)][4] - vₖ_des)^2 + α₂ * c_coll(β)
		else 
			return u[3:end]' * R * u[3:end] + α₁ * (β[Int(length(β)//2 + 4)] - vₖ_des)^2 + α₂ * c_coll(β)
		end
	end

	function cₗ²(β)
		# return 0.0
		if typeof(β) == BlockVector
			return α₁ * norm(β[Block(2)][4] - vₖ_des, 2)^2 + α₂ * c_coll(β)
		else
			return α₁ * norm(β[Int(length(β)//2 + 4)] - vₖ_des, 2)^2 + α₂ * c_coll(β)
		end
	end

	env = init_base_environment(;
			state_dynamics = state_dynamics,
			observation_function = observation_function,
			num_agents = 2,
			state_dim = 4,
		    dynamics_noise_dim = 4,
			observation_noise_dim = 4,
			initial_state = initial_state,
			final_time = time_steps)

	players = [
		init_player(;
			player_type = type_SDGiBS,
			player_id = 1,
			belief = initial_beliefs[Block(1)],
			cost = cₖ¹,
			final_cost = cₗ¹,
			action_space = 2,
			observation_space = 4,
			default_action = [0.0, 0.0],# accel, steer
			time = env.final_time,
			num_players = 2),
		init_player(;
			player_type = type_SDGiBS,
			player_id = 2,
			belief = initial_beliefs[Block(2)],
			cost = cₖ²,
			final_cost = cₗ²,
			action_space = 2,
			observation_space = 4,
			default_action = [0.0, 0.0],# accel, steer
			time = env.final_time,
			num_players = 2)
		]

	return surveillance_demo(env, players)
end

# Code from: https://discourse.julialang.org/t/plot-ellipse-in-makie/82814/2
function getellipsepoints(cx, cy, rx, ry, θ)
	t = range(0, 2*pi, length=100)
	ellipse_x_r = @. rx * cos(t)
	ellipse_y_r = @. ry * sin(t)
	R = [cos(θ) sin(θ); -sin(θ) cos(θ)]
	r_ellipse = [ellipse_x_r ellipse_y_r] * R
	x = @. cx + r_ellipse[:,1]
	y = @. cy + r_ellipse[:,2]
	[Point2f(t) for t in zip(x, y)]
end