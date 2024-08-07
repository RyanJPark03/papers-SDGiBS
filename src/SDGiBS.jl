module SDGiBS

using BlockArrays
using LinearAlgebra
using Distributions
using GLMakie

using ForwardDiff, DiffResults
using ForwardDiff: Chunk, JacobianConfig, HessianConfig

include("FiniteDiff.jl")
# include("../experiments/Players.jl")
# include("AbstractPlayer.jl")
# include("SDGiBSPlayer.jl")

"""
	Used to update the belief states of an array of players given their observations and their actions

	params:
		env::Base_Environment - the environment in which the players are playing
		players::Array{Player} - the players in the game
		observations<:Vector - the observations of the players
			Must be in the same order as ``players``, i.e. the first observation must
			be the observation of the first player
		actions<:Vector - the actions of the players
			Must be in the same order as ``players``, i.e. the first action must
			be the action of the first player

	returns: BlockVector - the updated belief states of the players stacked on each other
"""

export belief_update
function belief_update(env, players::Array, observations, actions) 
	return calculate_belief_variables(env, players, observations, env.time, nothing, actions)[1]
end

"""
	Used to solve for the predicted actions of the players.

	params:
		players::Array{Player} - the players in the game
		env::Base_Environment - the environment in which the players are playing
		get_action::Function - the function used to get the action of a player
			signature looks like (players::Array{Player}, playe_index::Int, time::Int;
									state::Vector || BlockVector = belief_state)
		horizon::Int - the planning horizon
		μᵦ::Float64 - the regularization parameter for belief. 
			Penalizes deviations from the nominal beliefs
		μᵤ::Float64 - the regularization parameter for control.
			Penalizes large deviations from nominal controls
		ϵ::Float64 - the convergence threshold for the solver
	
	returns:
		Tuple{BlockVector, BlockVector, Array{Function}}
		- the predicted beliefs of the players, b̄
		- the predicted actions of the players, ū
		- the feedback law of the players, π

"""

export SDGiBS_solve_action
function SDGiBS_solve_action(players::Array, env, action_selector; horizon = 1, μᵦₒ = 1.0, μᵤₒ = 1.0, ϵ = 1e-4)
	println("calling solver ...")

	μᵦ = [copy(μᵦₒ) for _ in 1:length(players)]
	μᵤ = [copy(μᵤₒ) for _ in 1:length(players)]
	# fig = Figure()
	solver_iter_solutions = []

	# Convenience variables
	N = env.num_agents
	action_space = [player.action_space for player in players]
	ηᵤ = sum(action_space)
	η_z	 = sum([player.observation_space for player in players])
	η_zz = sum([player.observation_space^2 for player in players])
	ηₓ = env.state_dim * N
	ηₓₓ = env.state_dim^2 * N
	c = [player.final_cost for player in players]
	ck = [(x_u) -> player.cost(x_u[1:ηₓ + ηₓₓ], x_u[ηₓ + ηₓₓ+1:end]) for player in players]
	

	# Preallocations: # TODO: Make all the functions in-place 
	V = [0.0 for _ in 1:N]
	V_b = [zeros(ηₓ + ηₓₓ) for _ in 1:N]
	V_bb = [zeros((ηₓ + ηₓₓ, ηₓ + ηₓₓ)) for _ in 1:N]
	x_u = zeros(ηₓ + ηₓₓ + ηᵤ)
	Wₖ = zeros((ηₓ + ηₓₓ, ηₓ))
	Wₛ = zeros((ηₓ + ηₓₓ, ηₓ, ηₓ + ηᵤ))
	gₛ = zeros((ηₓ + ηₓₓ, ηₓ + ηᵤ))
	tt_idx = 0
	Qⁱ = zeros(N)
	Qₛⁱ = [zeros(ηₓ + ηₓₓ) for _ in 1:N]
	Qₛₛⁱ = [zeros((ηₓ + ηₓₓ, ηₓ + ηₓₓ)) for _ in 1:N]
	Q_bⁱ = [zeros(ηₓ + ηₓₓ) for _ in 1:N]
	Q_bbⁱ = [zeros((ηₓ + ηₓₓ, ηₓ + ηₓₓ)) for _ in 1:N]
	Q̂_u = zeros(ηᵤ)
	Q_uⁱ = [zeros(ηᵤ) for _ in 1:N]
	Q̂_uu = zeros((ηᵤ, ηᵤ))
	Q_uuⁱ = [zeros((ηᵤ, ηᵤ)) for _ in 1:N]
	Q̂_ub = zeros((ηᵤ, ηₓ + ηₓₓ))
	Q_ubⁱ = [zeros((ηᵤ, ηₓ + ηₓₓ)) for _ in 1:N]
	jₖ = zeros(ηᵤ)
	Kₖ = zeros((ηᵤ, ηₓ + ηₓₓ))


	# Find the actual planning horizon and final time 
	#	(since horizon may extend past simulation's time interval)
	final_planning_time = min(env.time + horizon - 1, env.final_time)
	actual_horizon = final_planning_time - env.time

	# Initialize covariance matrix
	# 	TODO: this will scale n^3 with number of players in Second Order Formulation...
	Σₒ = BlockArray{Float64}(undef, [env.state_dim for player in players],
		[env.state_dim for player in players])
	for ii in eachindex(players)
		Σₒ[Block(ii, ii)] .= reshape(players[ii].belief[env.state_dim+1:end], tuple(env.state_dim, env.state_dim))
	end

	# Setup convenience functions to access player's predicted actions
	total_feedback_law = (tt, belief_state) -> vcat([action_selector(players, ii, tt; state = belief_state) for ii in eachindex(players)]...)
	u_k = (tt, belief_state) -> (tt == horizon) ? BlockVector([0.0 for _ in 1:ηᵤ], action_space) :
		total_feedback_law(tt, belief_state) 

	# Initial nominal trajectory
	b̄, ū, _ = simulate(env, players, u_k, nothing, env.time, final_planning_time)
	push!(solver_iter_solutions, get_plottables(b̄, ū))


	# Sanity check
	@assert length(b̄) == length(ū) + 1
	@assert length(b̄) == actual_horizon + 1

	# Iteration variables
	Q_old = [Inf for _ in eachindex(players)]
	Q_new = cost(players, b̄, ū)
	deltaQ = Q_new - Q_old
	cost_vars = DiffResults.HessianResult(x_u)
	iter = 0
	set_new_iter = false

	# initialize final answer
	π::Array{Function} = [(belief_state) -> u_k(tt, belief_state) for tt in 1:actual_horizon]

	# Functions to grab matrix form of belief update
	#	bₖ₊₁ ≈ gₖ + Wₖ * Εₖ, where Εₖ ~ N(0, I)
	W = (x) -> calculate_matrix_belief_variables(x[1:ηₓ + ηₓₓ], x[ηₓ + ηₓₓ+1:end]; env = env, players = players)[1]
	g = (x) -> calculate_matrix_belief_variables(x[1:ηₓ + ηₓₓ], x[ηₓ + ηₓₓ+1:end]; env = env, players = players, calc_W = false)[2]

	while norm(deltaQ, 2) > ϵ
		set_new_iter = false
		# Backward Pass
		println("iter: ", iter)
		iter += 1

		# Value function termination conditions
		V .= map((cᵢ) -> cᵢ(b̄[end]), c)
		V_b .= map((cᵢ) -> ForwardDiff.gradient(cᵢ, b̄[end]), c)
		V_bb .= map((cᵢ) -> ForwardDiff.hessian(cᵢ, b̄[end][1:end]), c)
		# println("V_bb")
		# for i in eachindex(V_bb)
		# 	show(stdout, "text/plain", V_bb[i])
		# 	println()
		# end
		# println()

		for tt in final_planning_time-1:-1:env.time
			tt_idx = tt - env.time + 1

			# Construct concatenated state-action vector
			println("planning at tt: ", tt, "converted into tt_idx: ", tt_idx)
			x_u .= vcat(b̄[tt_idx][1:end], π[tt_idx](b̄[tt_idx])[1:end])

			# Calculate Wₖ and gₖ # TODO: in place
			Wₖ = W(x_u)
			Wₛ = finite_diff(W, x_u)
			gₛ = ForwardDiff.jacobian(g, x_u)

			for ii in 1:N
				cost_vars = ForwardDiff.hessian!(cost_vars, ck[ii], x_u)
				# Main.@infiltrate imag(DiffResults.value(cost_vars) + V[ii] + 0.5 * sum([Wₖ[1:end, j]' * V_bb[ii] * Wₖ[1:end, j] for j in 1:ηₓ])) != 0.0
				Qⁱ[ii] = DiffResults.value(cost_vars) + V[ii] +
					0.5 * sum([Wₖ[1:end, j]' * V_bb[ii] * Wₖ[1:end, j] for j in 1:ηₓ])
				Qₛⁱ[ii] = DiffResults.gradient(cost_vars) + gₛ' * V_b[ii] +
					sum([Wₛ[:, j, :]' * V_bb[ii] * Wₖ[:, j] for j in 1:ηₓ])
				# Belief regularizatin: (V_bb[ii] + μ * I) instead of V_bb[ii]
				Qₛₛⁱ[ii] = DiffResults.hessian(cost_vars) + gₛ' * (V_bb[ii] + μᵦ[ii] * I) * gₛ +
					sum([Wₛ[:, j, :]' * (V_bb[ii] + μᵦ[ii] * I) * Wₛ[:, j, :] for j in 1:ηₓ])
					
				prev_action_spaces = sum(action_space[1:ii - 1])
				prev_and_cur_action_spaces = sum(action_space[1:ii])

				Q̂_u[prev_action_spaces+1:prev_and_cur_action_spaces] =
					Qₛⁱ[ii][ηₓ+ηₓₓ+prev_action_spaces+1:ηₓ+ηₓₓ+prev_and_cur_action_spaces, :]
				Q̂_ub[prev_action_spaces+1:prev_and_cur_action_spaces, :] = 
					Qₛₛⁱ[ii][ηₓ+ηₓₓ+prev_action_spaces+1:ηₓ+ηₓₓ+prev_and_cur_action_spaces, 1:ηₓ+ηₓₓ]
				
				Q_bⁱ[ii] = vec(Qₛⁱ[ii][1:ηₓ + ηₓₓ, :])
				Q_bbⁱ[ii] = Qₛₛⁱ[ii][1:ηₓ + ηₓₓ, 1:ηₓ + ηₓₓ]
				Q_uⁱ[ii] = vec(Qₛⁱ[ii][ηₓ + ηₓₓ + 1:end, :])
				
				Q_ubⁱ[ii] = Qₛₛⁱ[ii][ηₓ + ηₓₓ + 1:end, 1:ηₓ + ηₓₓ]

				# Control regularization
				Qₛₛⁱ[ii] += μᵤ[ii] * I
				Q_uuⁱ[ii] = Qₛₛⁱ[ii][ηₓ + ηₓₓ + 1:end, ηₓ + ηₓₓ + 1:end]
				println("after player ", ii, "'s backpass at time: ", tt)
				Q̂_uu[prev_action_spaces+1:prev_and_cur_action_spaces, :] = 
					Qₛₛⁱ[ii][ηₓ+ηₓₓ+prev_action_spaces+1:ηₓ+ηₓₓ+prev_and_cur_action_spaces, ηₓ+ηₓₓ+1:end]
				show(stdout, "text/plain", Q̂_uu)
				println()
				println("Qub:")
				show(stdout, "text/plain", Q̂_ub)
				println()
				println("Qu")
				show(stdout, "text/plain", Q̂_u)
				println()
			end
			# pseudo_inverse = pinv(Q̂_uu)
			# jₖ .= - pseudo_inverse * Q̂_u
			# Kₖ .= - pseudo_inverse * Q̂_ub # overloaded notation, Kₖ has a different value in belief update
			jₖ .= - Q̂_uu \ Q̂_u
			Kₖ .= - Q̂_uu \ Q̂_ub # overloaded notation, Kₖ has a different value in belief update
			# println("Kₖ: ")
			# show(stdout, "text/plain", Kₖ)
			# println()
			π[tt_idx] = create_policy(ū[tt_idx], jₖ, Kₖ)

			# Backwards iteration of value function
			for ii in 1:N
				V[ii] = Qⁱ[ii] + (Q_uⁱ[ii]'*jₖ)[1, 1] + (0.5*jₖ'*Q_uuⁱ[ii]*jₖ)[1, 1]
				V_b[ii] .= Q_bⁱ[ii] + Kₖ' * Q_uuⁱ[ii] * jₖ + Kₖ' * Q_uⁱ[ii] + Q_ubⁱ[ii]' * jₖ
				V_bb[ii] .= Q_bbⁱ[ii] + Kₖ' * Q_uuⁱ[ii] * Kₖ + Kₖ' * Q_ubⁱ[ii] + Q_ubⁱ[ii]' * Kₖ
			end
			
		end
		# Forwards Pass

		b̄_new, ū_new, _ = simulate(env, players, π, b̄, env.time, final_planning_time; noise=false)
		push!(solver_iter_solutions, get_plottables(b̄_new, ū_new))
		println("solving .... new ū:")
		show(stdout, "text/plain", ū_new)
		println()
		
		Q_new = cost(players, b̄_new, ū_new)
		println("Q_new: ", Q_new, " Q_old: ", Q_old, "\n\t delta = ", Q_new - Q_old)
		println("μᵤ: ", μᵤ, " μᵦ: ", μᵦ)

		for ii in eachindex(players)
			if Q_new[ii] > Q_old[ii]
				println("increasing regularization for player ", ii)
				μᵤ[ii] *= 1.5
				μᵦ[ii] *= 1.5
			else
				println("decreasing regularization for player ", ii)
				if !set_new_iter# Don't want other players to change iteration variables if previous players have already done so
					set_new_iter = true
					println("setting new")
					deltaQ = Q_new - Q_old
					Q_old = Q_new
					b̄ = b̄_new
					ū = ū_new
				end
				μᵤ[ii] /= 1.5
				μᵦ[ii] /= 1.5
				
			end
		end
		if any([μᵤ[ii] > 1e10 || μᵦ[ii] > 1e10 || μᵤ[ii] < 1e-10 || μᵦ[ii] < 1e-10 for ii in eachindex(players)])
			plot_error(solver_iter_solutions)
			error("did not converge...")
			break
		end
	end
	println("solver ran for ", iter, " iterations")
	println("\tdeltaQ: ", deltaQ,"\n\tQ_new: ", Q_new, "\n\tQ_old: ", Q_old)
	println("\tdeltaQ norm: ", norm(deltaQ, 2), ", ϵ = ", ϵ, ", iter: ", iter)
	push!(env.solver_history, solver_iter_solutions)
	return b̄, ū, π
end

function create_policy(nominal_control, feed_forward, feed_backward)
	function (δb)
		return nominal_control + 1.0 * (vec(feed_forward) + feed_backward * δb)
	end
end

function simulate(env, players, ū, b̄, time, end_time; noise = false)
	b̄_new = [BlockVector{Float64}(undef, [env.state_dim + env.state_dim^2 for _ in eachindex(players)]) for _ in time:end_time]
	sts = [BlockVector{Float64}(undef, [env.state_dim for _ in eachindex(players)]) for _ in time:end_time]
	ū_actual = [BlockVector{Float64}(undef, [player.action_space for player in players]) for _ in time:end_time-1]

	belief_length = length(players[1].belief[Block(1)]) # TODO: make adjustable per player
	b̄_new[1] = BlockVector(vcat([players[ii].history[time][3] for ii in eachindex(players)]...),
		[belief_length for player in players])
	sts[1] .= env.current_state

	dynamics_noise = BlockVector(zeros(env.dynamics_noise_dim * length(players)), [env.dynamics_noise_dim for _ in 1:length(players)])
	observation_noise = BlockVector(zeros(env.observation_noise_dim * length(players)), [env.observation_noise_dim for _ in 1:length(players)])

	for tt in time:end_time-1
		println("simulating time: ", tt)
		if isa(ū, Function)
			if isnothing(b̄) # first rollout does not have a nominal trajectory
				ū_actual[tt-time+1] .= vcat([player.history[end][1] for player in players]...)
			else
				ū_actual[tt-time+1] .= ū(tt, δb = b̄_new[tt-time+1] - b̄[tt-time+1])
			end
		elseif typeof(ū) == Vector{Function}
			println("\tb's: ", b̄_new[tt-time+1][1:20], "\n\t",  b̄_new[tt-time+1][21:end],"\n\t", b̄[tt-time+1][1:20], "\n\t", b̄[tt-time+1][21:end])
			δb = b̄_new[tt-time+1] - b̄[tt-time+1]
			println("\tδb = ", δb[1:20])
			println("\t", δb[21:end])
			ū_actual[tt-time+1] .= ū[tt-time+1](b̄_new[tt-time+1] - b̄[tt-time+1])
		end
		println("\tu_acutal: ", ū_actual[tt-time+1])

		if noise 
			dynamics_noise .= [rand(Distributions.Normal()) for _ in 1:env.dynamics_noise_dim*length(players)]
			observation_noise .= [rand(Distributions.Normal()) for _ in 1:env.observation_noise_dim*length(players)]
		end
		sts[tt-time+2] .= env.state_dynamics(sts[tt-time+1], ū_actual[tt-time+1], dynamics_noise)
		observations = env.observation_function(states = sts[tt-time+2], m = observation_noise)
		println("\tnew state: ", sts[tt-time+2])
		println("\tobservations: ", observations)
		β, Aₖ, Mₖ, Hₖ, Nₖ, Kₖ, x̂ₖ₊₁, Σₖ₊₁ = calculate_belief_variables(env, players, observations, tt, b̄_new[tt-time+1], ū_actual[tt-time+1])

		b̄_new[tt-time+2] .= β
	end

	return b̄_new, ū_actual, sts
end

function cost(players, b, u)
	cₖ = [player.cost for player in players]
	cₗ = [player.final_cost for player in players]

	Q = [0.0 for _ in eachindex(players)]

	for ii in eachindex(players)
		for tt in 1:length(b)-1
			Q[ii] += cₖ[ii](b[tt], u[tt])
		end
		Q[ii] += cₗ[ii](b[end])
	end
	return Q
end

function get_prefixes(b̄, prefix_length::Int)
	return BlockVector(vcat([b̄[Block(ii)][1:prefix_length] for ii in eachindex(blocks(b̄))]...),
		[prefix_length for _ in eachindex(blocks(b̄))])
end

function calculate_matrix_belief_variables(β, u; env, players, calc_W = true)
	num_players = length(players)
	mean_lengths = [player.observation_space for player in players]
	belief_lengths = [mean_lengths[ii] + mean_lengths[ii]^2 for ii in eachindex(players)]

	x̂ₖ = vcat([β[sum(belief_lengths[1:ii-1])+1:sum(belief_lengths[1:ii])][1:env.state_dim] for ii in 1:num_players]...)
	Σₖ_temp = BlockArray{Real}(undef, mean_lengths, mean_lengths)
	for ii in eachindex(players)
		for jj in eachindex(players)
			if ii == jj
				Σₖ_temp[Block(ii, jj)] .= reshape(β[sum(belief_lengths[1:ii-1])+1:sum(belief_lengths[1:ii])][mean_lengths[ii]+1:end], tuple(mean_lengths...))
			else
				Σₖ_temp[Block(ii, jj)] .= zeros((mean_lengths[ii], mean_lengths[jj]))
			end
		end
	end
	Σₖ = Matrix(Σₖ_temp)

	uₖ = u
	m = [0.0 for _ in 1:env.dynamics_noise_dim*num_players]
	n = [0.0 for _ in 1:env.observation_noise_dim*num_players]

	f = (x) -> env.state_dynamics(
			BlockVector(x[1:length(x̂ₖ)], mean_lengths),
			BlockVector(x[length(x̂ₖ)+1:length(x̂ₖ)+length(uₖ)], [player.action_space for player in players]),
			BlockArray(x[length(x̂ₖ)+length(uₖ)+1:end], [env.dynamics_noise_dim for _ in 1:num_players]))

	h = (x) -> env.observation_function(
		states = BlockVector(x[1:length(x̂ₖ)], mean_lengths),
		m = BlockVector(x[length(x̂ₖ)+1:end], [env.observation_noise_dim for _ in 1:num_players]))


	f_jacobian = ForwardDiff.jacobian(f, BlockVector(vcat([x̂ₖ, uₖ, m]...), [length(x̂ₖ), length(uₖ), length(m)]))
	Aₖ = f_jacobian[:, 1:length(x̂ₖ)]
	Mₖ = f_jacobian[:, length(x̂ₖ)+length(uₖ)+1:end]
	h_jacobian = ForwardDiff.jacobian(h, BlockVector(vcat([x̂ₖ, n]...), [length(x̂ₖ), length(n)]))
	Hₖ = h_jacobian[:, 1:length(x̂ₖ)]
	Nₖ = h_jacobian[:, length(x̂ₖ)+1:end]


	Γₖ₊₁ = Aₖ * Σₖ * Aₖ' + Mₖ * Mₖ'
	Kₖ = Γₖ₊₁ * Hₖ' * ((Hₖ * Γₖ₊₁ * Hₖ' + Nₖ * Nₖ') \ I)

	noiseless_x̂ₖ₊₁ = env.state_dynamics(x̂ₖ, uₖ, m)
	temp = BlockArray(Γₖ₊₁ - Kₖ * Hₖ * Γₖ₊₁, mean_lengths, mean_lengths)
	covs = [Matrix(temp[Block(ii, ii)]) for ii in eachindex(players)]
	means = [noiseless_x̂ₖ₊₁[sum(mean_lengths[1:ii-1])+1:sum(mean_lengths[1:ii])] for ii in eachindex(players)]
	g = vcat(vcat(means...), vcat([vec(covs[ii]) for ii in eachindex(covs)]...))
	if calc_W
		W = vcat(sqrt(Kₖ * Hₖ * Γₖ₊₁), zeros((sum(mean_lengths .^ 2), sum(mean_lengths))))
		Σ_new = (noise) -> g + W * (noise)
		x_new = env.state_dynamics(x̂ₖ, uₖ, BlockVector([0.0 for _ in 1:env.dynamics_noise_dim*num_players],
					[env.dynamics_noise_dim for _ in 1:num_players]))
	else
		W = nothing
		Σ_new = nothing
		x_new = nothing
	end

	return W, g, Aₖ, Mₖ, Hₖ, Nₖ, Kₖ, Σ_new, x_new
end

function calculate_belief_variables(env, players, observations, time, β, u_k)
	num_players = length(players)
	if isnothing(β)
		β = BlockVector(vcat([player.belief for player in players]...), [length(player.belief) for player in players])
	end

	mean_lengths = [env.state_dim for _ in 1:num_players]
	cov_length = Int(sqrt(Int(length(β) // num_players - env.state_dim)))
	cov_lengths = [cov_length for _ in 1:num_players]

	x̂ₖ = BlockVector(vcat([β[Block(ii)][1:env.state_dim] for ii in 1:num_players]...), mean_lengths)
	Σₖ = BlockArray{Float64}(undef, cov_lengths, cov_lengths)

	for ii in eachindex(players)
		for jj in eachindex(players)
			if ii == jj
				Σₖ[Block(ii, jj)] .= reshape(β[Block(ii)][env.state_dim+1:end], tuple(cov_lengths...))
			else
				Σₖ[Block(ii, jj)] .= zeros((cov_length, cov_length))
			end
		end
	end
	
	if isnothing(u_k)
		uₖ = BlockVector(vcat([player.predicted_control[time - env.time + 1][Block(player.player_id)] for player in players]...),
			[player.action_space for player in players])
	else
		uₖ = u_k
	end
	m = BlockVector([0.0 for _ in 1:env.dynamics_noise_dim*num_players],
		[env.dynamics_noise_dim for _ in 1:num_players])
	n = BlockVector([0.0 for _ in 1:env.observation_noise_dim*num_players],
		[env.observation_noise_dim for _ in 1:num_players])

	f = (x) -> env.state_dynamics(
		BlockVector(x[Block(1)], mean_lengths),
		BlockVector(x[Block(2)], [player.action_space for player in players]),
		BlockArray(x[Block(3)], [env.dynamics_noise_dim for _ in 1:num_players]))

	h = (x) -> env.observation_function(
		states = BlockVector(x[Block(1)], mean_lengths),
		m = BlockVector(x[Block(2)], [env.observation_noise_dim for _ in 1:num_players]))

	# println("gradient at:\n\t\tx̂ₖ: ", round.(x̂ₖ, digits = 5),"\n\t\tûₖ: ", uₖ,"\n\t\tmₖ: ", m)

	j_cfg = JacobianConfig(f, BlockVector(vcat([x̂ₖ, uₖ, m]...), [length(x̂ₖ), length(uₖ), length(m)]), Chunk{20}())
	f_jacobian = ForwardDiff.jacobian(f, BlockVector(vcat([x̂ₖ, uₖ, m]...), [length(x̂ₖ), length(uₖ), length(m)]), j_cfg)
	Aₖ = round.(f_jacobian[:, 1:length(x̂ₖ)], digits = 100)
	Mₖ = round.(f_jacobian[:, length(x̂ₖ)+length(uₖ)+1:end], digits = 100)

	h_jacobian = ForwardDiff.jacobian(h, BlockVector(vcat([x̂ₖ, n]...), [length(x̂ₖ), length(n)]))

	Hₖ = round.(h_jacobian[:, 1:length(x̂ₖ)], digits = 100)
	Nₖ = round.(h_jacobian[:, length(x̂ₖ)+1:end], digits = 100)

	Γₖ₊₁ = Aₖ * Σₖ * Aₖ' + Mₖ * Mₖ'
	# Main.@infiltrate any(isnan.(Hₖ * Γₖ₊₁ * Hₖ' + Nₖ * Nₖ'))
	Kₖ = Γₖ₊₁ * Hₖ' * (round.(Hₖ * Γₖ₊₁ * Hₖ' + Nₖ * Nₖ', digits = 100) \ I)

	noiseless_x̂ₖ₊₁ = env.state_dynamics(x̂ₖ, uₖ, m)

	temp  = env.observation_function(states = noiseless_x̂ₖ₊₁, m = n)
	x̂ₖ₊₁ = noiseless_x̂ₖ₊₁ + Kₖ * (observations - temp)
	Σₖ₊₁ = Γₖ₊₁ - Kₖ * Hₖ * Γₖ₊₁

	x̂_temp = BlockVector(x̂ₖ₊₁, mean_lengths)
	Σ_block = BlockArray(Σₖ₊₁, cov_lengths, cov_lengths)
	temp = vcat([Σ_block[Block(ii, ii)] for ii in eachindex(players)]...)
	Σ_temp = BlockVector(vec(temp'), [cov_length^2 for _ in eachindex(players)])

	β_new = BlockVector(vcat([vcat(x̂_temp[Block(ii)], Σ_temp[Block(ii)]) for ii in eachindex(players)]...),
		[mean_lengths[i] + cov_lengths[i]^2 for i in eachindex(players)])
	return β_new, Aₖ, Mₖ, Hₖ, Nₖ, Kₖ, x̂_temp, Σ_block
end


function get_plottables(b̄, ū)
	player1_x1 = [b[Block(1)][1] for b in b̄]
	player1_x2 = [b[Block(1)][2] for b in b̄]
	player1_covs = [reshape(b[Block(1)][5:end], (4, 4)) for b in b̄]
	player1_radii = [(cov[1, 1], cov[2, 2]) for cov in player1_covs]
	player1_elipses = [getellipsepoints(player1_x1[i], player1_x2[i], player1_radii[i][1], player1_radii[i][2], 0.0) for i in eachindex(player1_x1)]

	player2_x1 = [b[Block(2)][1] for b in b̄]
	player2_x2 = [b[Block(2)][2] for b in b̄]
	player2_covs = [reshape(b[Block(2)][5:end], (4, 4)) for b in b̄]
	player2_radii = [(cov[1, 1], cov[2, 2]) for cov in player2_covs]
	player2_elipses = [getellipsepoints(player2_x1[i], player2_x2[i], player2_radii[i][1], player2_radii[i][2], 0.0) for i in eachindex(player2_x1)]

	return (; p1x = player1_x1, p1y = player1_x2, p1e = player1_elipses,
		p2x = player2_x1, p2y = player2_x2, p2e = player2_elipses)
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

function plot_error(plottables)
	fig = Figure()
	ax = Axis(fig[1, 1], xlabel = "x", ylabel = "y")
	sg = SliderGrid(
        fig[2, 1],
        (label = "solver iteration", range = 1:length(plottables), format = x-> "", startvalue = 1)
    )
    solver_iteration = lift(sg.sliders[1].value) do a
        Int(a)
    end
	player_locations = lift(solver_iteration) do a
		# time_slices = [get_history(player, a) for player in demo.players]
		time_slice = plottables[a]

		player_1_point = [Point2f(time_slice.p1x[i], time_slice.p1y[i]) for i in eachindex(time_slice.p1x)]
		player_2_point = [Point2f(time_slice.p2x[i], time_slice.p2y[i]) for i in eachindex(time_slice.p2x)]
		return player_1_point, player_2_point, time_slice.p1e[end], time_slice.p2e[end]
	end

	p1p = @lift $(player_locations)[1]
	p2p = @lift $(player_locations)[2]
	p1e = @lift $(player_locations)[3]
	p2e = @lift $(player_locations)[4]

	scatterlines!(ax, p1p; color = :blue)
	scatterlines!(ax, p2p; color = :red)
	lines!(ax, p1e; color = :black)
	lines!(ax, p2e; color = :black)
	display(fig)
	return fig
end
end