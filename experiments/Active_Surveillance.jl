"""
x⁽ⁱ⁾ = [x⁽ⁱ⁾, y⁽ⁱ⁾, θ⁽ⁱ⁾, v⁽ⁱ⁾]
u⁽ⁱ⁾ = [u_acceleration⁽ⁱ⁾, u_steer⁽ⁱ⁾]
"""


using BlockArrays
using LinearAlgebra
using Distributions
using GLMakie
using ProgressBars

include("Environment.jl")
include("Players.jl")


struct surveillance_demo{}
	env::base_environment
	players::Array{Player}
end


function active_surveillance_demo_main()
	open("./out.temp", "w") do file
		redirect_stdout(file) do 
			run_active_surveillance_demo(30, .1)
		end
	end
end

function run_active_surveillance_demo(time_steps, τ)
	surveillance_center = [0, 0]
	surveillance_radius = 10

	demo = init(time_steps, τ; surveillance_center = surveillance_center, surveillance_radius = surveillance_radius,
	L = 1)
    trajectory = []
    push!(trajectory, demo.env.current_state)

    for tt in ProgressBar(1:demo.env.final_time - 1)
		time_step_all_coop(demo.players, demo.env)
        push!(trajectory, demo.env.current_state)
    end
	# Main.@infiltrate
	coords1 = [[x[1] for x in trajectory], [x[2] for x in trajectory]]
	coords2 = [[x[5] for x in trajectory], [x[6] for x in trajectory]]

	belief_coords1 = [[x[3][1] for x in demo.players[1].history], [x[3][2] for x in demo.players[1].history]]
	belief_coords2 = [[x[3][1] for x in demo.players[2].history], [x[3][2] for x in demo.players[2].history]]

	# cov1 = # TODO: plot cov

	observations1 = [[demo.players[1].history[x][2][1] for x in 2:length(demo.players[1].history)],
	[demo.players[1].history[x][2][2] for x in 2:length(demo.players[1].history)]]
	observations2 = [[demo.players[2].history[x][2][1] for x in 2:length(demo.players[1].history)],
	[demo.players[2].history[x][2][2] for x in 2:length(demo.players[1].history)]]



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
		action = [slice[2] for slice in time_slices]
		println("p2 actions: ")
		display(action[2])
		beliefs = [time_slice[3] for time_slice in time_slices]

		positions = [belief[1:2] for belief in beliefs]
		covs = [reshape(belief[5:end], (4, 4)) for belief in beliefs]
		display(covs[1])
		display(covs[2])
		radii = [(cov[1, 1], cov[2, 2]) for cov in covs]
		display(radii)
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
	return fig
end

function init(time_steps, τₒ; surveillance_center = [0, 0], surveillance_radius::Int = 10, 
	L::Int = 1)
	# τₒ = .1
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
			# new_state[4 * (i - 1) + 1 : 4 * i] .= states[4 * (i - 1) + 1 : 4 * i] + τ * ẋ + 0 .*  m[4 * (i - 1) + 1 : 4 * i]
			# new_state[4 * (i - 1) + 3] %= 2π
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

			# println("steer: ", steer) #steer is going to infinity
			steer = max(5π/6, min(-5π/6, steer))
			δv = (abs(steer) < ϵₛ) ? 0.0 : v / (L * tan(steer))
			ẋ = [v * cos(θ), v * sin(θ), δv, accel]# assign 2 5 for drawing

			# M scales motion noise mₖ according to size of u[i], i.e. more noise the bigger the control
			if block
				# new_state[Block(i)] .= states[Block(i)] + τ * ẋ + 0.0 .* m[Block(i)]
				new_state[Block(i)] .= states[Block(i)] + τ * ẋ + M(u[Block(i)]) * m[Block(i)]
				new_state[Block(i)][3] %= 2π
			else
				new_state[4 * (i - 1) + 1 : 4 * i] .= states[Block(i)] + τ * ẋ + M(u[Block(i)]) * m[Block(i)]
				# new_state[4 * (i - 1) + 1 : 4 * i] .= states[Block(i)] + τ * ẋ + 0.0 .* m[Block(i)]
				new_state[4 * (i - 1) + 3] %= 2π
			end
		end
		return new_state
	end

	function measurement_noise_scaler1(state::Vector; surveillance_center = [0, 0], surveillance_radius::Int = 10)
		# Only take x and y coords from state vector
		n_outer = .05 * (norm(state[1:2] - surveillance_center, 2) - surveillance_radius)^2
		n_outer = max(0.01, n_outer) # make sure noise multiplier doesn't get too small, don't want players to be able to see each other perfectly
		n_outer = min(1000, n_outer) # make sure noise multiplier doesn't get too large
		n = n_outer

        v = .2 * state[4]^2 # velocity scaled noise
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

	initial_state = BlockVector{Float64}(undef, [4 for _ in 1:2])
	initial_state[Block(1)] .= [10.0, 20.0, 0.0, 10.0] # Player 1, surveiller
	initial_state[Block(2)] .= [8.0, 15.0, 0.0, 10.0]

	env = init_base_environment(;
			state_dynamics = state_dynamics,
			observation_function = observation_function,
			num_agents = 2,
			state_dim = 4,
		    dynamics_noise_dim = 4,
			observation_noise_dim = 4,
			initial_state = initial_state,
			final_time = time_steps) # 15 if inital action is 0 0.5

	initial_beliefs = BlockVector{Float64}(undef, [20, 20])
	initial_cov_matrix = [
		5.0 0.0 0.0 0.0;
		0.0 5.0 0.0 0.0;
		0.0 0.0 0.006 0.0;
		0.0 0.0 0.0 0.5;
	]
	initial_beliefs[Block(1)] .= vcat(copy(initial_state[Block(1)]), vec(copy(initial_cov_matrix)))
	initial_beliefs[Block(2)] .= vcat(copy(initial_state[Block(2)]), vec(copy(initial_cov_matrix)))

	function cₖ¹(β, u)
		R = Matrix(100 * I, 2, 2)
		if typeof(u) == BlockVector
			return u[Block(1)]' * R * u[Block(1)]
		else
			return u[1:2]' * R * u[1:2]
		end
	end
	cost_mult = 10.0
	function cₗ¹(β)
		if typeof(β) == BlockVector
			return cost_mult * prod(diag(reshape(β[Block(2)][5:end], (4, 4))))
		else
			return cost_mult * prod(diag(reshape(β[Int(length(β)//2 + 5):end], (4, 4))))
		end
	end

	α₁ = 1.0
	α₂ = 10.0
	vₖ_des = initial_state[Block(2)][4]
	function c_coll(β)
		if typeof(β) == BlockVector
			return exp(-norm(β[Block(1)][1:2] - β[Block(2)][1:2], 2))
		else
			return exp(-norm(β[1:2] - β[Int(length(β)//2 + 1):Int(length(β)//2 + 2)], 2))
		end
	end
	function cₖ²(β::T, u::T) where T
		return 0.0
		R = Matrix(10 * I, 2, 2)
		if typeof(β) == BlockVector
			return u[Block(2)]' * R * u[Block(2)] + α₁ * (β[Block(2)][4] - vₖ_des)^2 + α₂ * c_coll(β)
		else 
			return u[3:end]' * R * u[3:end] + α₁ * (β[Int(length(β)//2 + 4)] - vₖ_des)^2 + α₂ * c_coll(β)
		end
	end

	function cₗ²(β)
		return 0.0
		if typeof(β) == BlockVector
			return α₁ * norm(β[Block(2)][4] - vₖ_des, 2)^2 + α₂ * c_coll(β)
		else
			return α₁ * norm(β[Int(length(β)//2 + 4)] - vₖ_des, 2)^2 + α₂ * c_coll(β)
		end
	end
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