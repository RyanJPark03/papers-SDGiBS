module SDGiBS

using BlockArrays
using LinearAlgebra
using Distributions

using ForwardDiff, DiffResults
using ForwardDiff: Chunk, JacobianConfig, HessianConfig

include("FiniteDiff.jl")

export belief_update
function belief_update(env, players::Array, observations)
	return calculate_belief_variables(env, players, observations, env.time, nothing, nothing)[1]
end

export SDGiBS_solve_action
function SDGiBS_solve_action(players::Array, env, action_selector; μᵦ = 1000.0, μᵤ = 1000.0, ϵ = 1e-10)
	println("calling solver ...")
	Σₒ = BlockArray{Float64}(undef, [env.state_dim for player in players], [env.state_dim for player in players])
	for ii in eachindex(players)
		player = players[ii]
		Σₒ[Block(ii, ii)] .= reshape(player.belief[player.observation_space+1:end], (player.observation_space, player.observation_space))
	end

	total_feedback_law = (tt, belief_state) -> vcat([action_selector(players, ii, tt; state = belief_state) for ii in eachindex(players)]...)
	u_k = (tt, belief_state) -> (tt == env.final_time) ? BlockVector([0.0 for _ in 1:sum(action_length)], action_length) :
		total_feedback_law(tt, belief_state)

	action_length = [player.action_space for player in players]

	b̄, ū, _ = simulate(env, players, u_k, nothing, env.time)
	belief_length = length(b̄[1])

	@assert length(b̄) == length(ū) + 1
	@assert length(b̄) == env.final_time - env.time + 1

	# Iteration variables
	Q_old = [Inf for _ in eachindex(players)]
	Q_new = cost(players, b̄, ū)

	# initialize final answer
	π::Array{Function} = [() -> zero(sum(action_length)) for _ in env.time:env.final_time-1]

	c = [player.final_cost for player in players]
	ck = [(x_u) -> player.cost(x_u[1:belief_length], x_u[belief_length+1:end]) for player in players]

	ηₓ = sum([player.observation_space for player in players])

	W = (x) -> calculate_matrix_belief_variables(x[1:belief_length], x[belief_length+1:end]; env = env, players = players)[1]
	g = (x) -> calculate_matrix_belief_variables(x[1:belief_length], x[belief_length+1:end]; env = env, players = players, calc_W = false)[2]
	x_u = vcat(b̄[end][1:end], u_k(env.final_time, b̄[end]))

	V = map((cᵢ) -> cᵢ(b̄[end]), c)
	V_b = map((cᵢ) -> ForwardDiff.gradient(cᵢ, b̄[end]), c)
	V_bb = map((cᵢ) -> ForwardDiff.hessian(cᵢ, b̄[end][1:end]), c)

	while norm(Q_new - Q_old, 2) > ϵ
		for tt in env.final_time-1:-1:env.time
			# πₖ = ūₖ + jₖ + Kₖ * δbₖ
			# jₖ = -Q̂⁻¹ᵤᵤ * Q̂ᵤ
			# Kₖ = -Q̂⁻¹ᵤᵤ * Q̂ᵤ\_b

			x_u .= vcat(b̄[tt - env.time + 1][1:end], u_k(tt - env.time + 1, b̄[tt - env.time + 1])[1:end])
			Wₖ = W(x_u)
			for ii in eachindex(players)
				cost_vars = DiffResults.HessianResult(x_u)
				cost_vars = ForwardDiff.hessian!(cost_vars, ck[ii], x_u)

				Wₛ = finite_diff(W, x_u)
				gₛ = ForwardDiff.jacobian(g, x_u)

				Q = DiffResults.value(cost_vars) + V[ii] +
					0.5 * sum([Wₖ[1:end, j]' * V_bb[ii] * Wₖ[1:end, j] for j in 1:env.state_dim])

				Qₛ = DiffResults.gradient(cost_vars) + gₛ' * V_b[ii] +
					 sum([Wₛ[:, j, :]' * V_bb[ii] * Wₖ[:, j] for j in 1:ηₓ])

				Q_b = Qₛ[1:belief_length, :]
				Q_u = Qₛ[belief_length+1:end, :] # skip elems wrt belief

				# Belief regularizatin: (V_bb[ii] + μ * I) instead of V_bb[ii]
				Qₛₛ = DiffResults.hessian(cost_vars) + gₛ' * (V_bb[ii] + μᵦ * I) * gₛ +
					  sum([Wₛ[:, j, :]' * (V_bb[ii] + μᵦ * I) * Wₛ[:, j, :] for j in 1:ηₓ])

				Q_bb = Qₛₛ[1:belief_length, 1:belief_length]
				# Control regularization
				Q_ub = Qₛₛ[belief_length+1:end, 1:belief_length]
				Q_uu = Qₛₛ[belief_length+1:end, belief_length+1:end] + μᵤ * I

				jₖ = -Q_uu \ Q_u
				Kₖ = -Q_uu \ Q_ub # overloaded notation, Kₖ has a different value in belief update
				π[tt-env.time+1] = create_policy(ū[tt-env.time+1], jₖ, Kₖ)

				# Backwards iteration of value function
				V[ii] = Q + (Q_u'*jₖ)[1, 1] + (0.5*jₖ'*Q_uu*jₖ)[1, 1]
				V_b[ii] .= Q_b + Kₖ' * Q_uu * jₖ + Kₖ' * Q_u + Q_ub' * jₖ
				V_bb[ii] = Q_bb + Kₖ' * Q_uu * Kₖ + Kₖ' * Q_ub + Q_ub' * Kₖ
			end
		end
		b̄_new, ū_new, _ = simulate(env, players, π, b̄, env.time; noise=true)
		Q_new = cost(players, b̄_new, ū_new)
		# Forwards Pass
		if Q_new <= Q_old
			Q_old = Q_new
			b̄ = b̄_new
			ū = ū_new
			μᵤ *= 0.75
			μᵦ *= 0.75
		else
			# println("increasing regularization")
			if μᵤ > 1e10 || μᵦ > 1e10
				# println("μ too large")
				break
			end
			μᵤ *= 10 * env.final_time
			μᵦ *= 10 * env.final_time
		end
	end
	return b̄, ū, π
end

function create_policy(nominal_control, feed_forward, feed_backward)
	function (δb)
		return nominal_control + .1 * (vec(feed_forward) + feed_backward * δb)
	end
end

function simulate(env, players, ū, b̄, time; noise = false)
	# TODO: its not env state dim but observation dim
	b̄_new = [BlockVector{Float64}(undef, [env.state_dim + env.state_dim^2 for _ in eachindex(players)]) for _ in time:env.final_time]
	sts = [BlockVector{Float64}(undef, [env.state_dim for _ in eachindex(players)]) for _ in time:env.final_time]
	ū_actual = [BlockVector{Float64}(undef, [player.action_space for player in players]) for _ in time:env.final_time-1]

	belief_length = length(players[1].belief[Block(1)]) # TODO: make adjustable per player
	b̄_new[1] = BlockVector(vcat([players[ii].history[time][3] for ii in eachindex(players)]...),
		[belief_length for player in players])
	sts[1] .= env.current_state

	dynamics_noise = BlockVector(zeros(env.dynamics_noise_dim * length(players)), [env.dynamics_noise_dim for _ in 1:length(players)])
	observation_noise = BlockVector(zeros(env.observation_noise_dim * length(players)), [env.observation_noise_dim for _ in 1:length(players)])

	for tt in time:env.final_time-1
		# println("simulating time: ", tt)
		if isa(ū, Function)
			if isnothing(b̄) # first rollout does not have a nominal trajectory
				ū_actual[tt-time+1] .= vcat([player.history[end][1] for player in players]...)
			else
				ū_actual[tt-time+1] .= ū(tt, δb = b̄_new[tt-time+1] - b̄[tt-time+1])
			end
		elseif typeof(ū) == Vector{Function}
			ū_actual[tt-time+1] .= ū[tt-time+1](b̄_new[tt-time+1] - b̄[tt-time+1])
		end

		if noise 
			dynamics_noise .= [rand(Distributions.Normal()) for _ in 1:env.dynamics_noise_dim*length(players)]
			observation_noise .= [rand(Distributions.Normal()) for _ in 1:env.observation_noise_dim*length(players)]
		end
		sts[tt-time+2] .= env.state_dynamics(sts[tt-time+1], ū_actual[tt-time+1], dynamics_noise)
		observations = env.observation_function(states = sts[tt-time+2], m = observation_noise)

		β, Aₖ, Mₖ, Hₖ, Nₖ, Kₖ, x̂ₖ₊₁, Σₖ₊₁ = calculate_belief_variables(env, players, observations, tt, b̄_new[tt-time+1], ū_actual[tt-time+1])
		# println("norms:\n\tAₖ: ", norm(Aₖ), "\n\tMₖ: ", norm(Mₖ), "\n\tHₖ: ", norm(Hₖ), "\n\tNₖ: ", norm(Nₖ), "\n\tKₖ: ", norm(Kₖ))
		# show(stdout, "text/plain", Mₖ)
		# println()

		b̄_new[tt-time+2] .= β
	end

	return b̄_new, ū_actual, sts
end

function cost(players, b, u)
	cₖ = [player.cost for player in players]
	cₗ = [player.final_cost for player in players]

	Q = [0.0 for _ in eachindex(players)]

	for ii in eachindex(players)
		for tt in 1:length(blocks(b))-1
			if typeof(u) == Vector
				actions = u[tt][Block(ii)]
			else
				actions = BlockVector(vcat([u[Block(ii, tt)] for ii in eachindex(players)]...),
					[player.action_space for player in players])
			end
			Q[ii] += cₖ[ii](b[tt], actions)
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
			BlockArray(x[length(x̂ₖ)+length(uₖ)+1:end], [env.dynamics_noise_dim for _ in 1:num_players]), block = false)

	h = (x) -> env.observation_function(
		states = BlockVector(x[1:length(x̂ₖ)], mean_lengths),
		m = BlockVector(x[length(x̂ₖ)+1:end], [env.observation_noise_dim for _ in 1:num_players]), block = false)


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
	# Main.@infiltrate 
	# Kₖ, Γ is not sym (but second block is sym)
	if calc_W
		W = vcat(sqrt(Kₖ * Hₖ * Γₖ₊₁), zeros((sum(mean_lengths .^ 2), sum(mean_lengths))))
	else
		W = nothing
	end

	return W, g, Aₖ, Mₖ, Hₖ, Nₖ, Kₖ
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
		BlockArray(x[Block(3)], [env.dynamics_noise_dim for _ in 1:num_players]), block = false)

	h = (x) -> env.observation_function(
		states = BlockVector(x[Block(1)], mean_lengths),
		m = BlockVector(x[Block(2)], [env.observation_noise_dim for _ in 1:num_players]), block = false)

	# println("gradient at:\n\t\tx̂ₖ: ", round.(x̂ₖ, digits = 5),"\n\t\tûₖ: ", uₖ,"\n\t\tmₖ: ", m)

	j_cfg = JacobianConfig(f, BlockVector(vcat([x̂ₖ, uₖ, m]...), [length(x̂ₖ), length(uₖ), length(m)]), Chunk{20}())
	f_jacobian = ForwardDiff.jacobian(f, BlockVector(vcat([x̂ₖ, uₖ, m]...), [length(x̂ₖ), length(uₖ), length(m)]), j_cfg)
	Aₖ = round.(f_jacobian[:, 1:length(x̂ₖ)], digits = 10)
	Mₖ = round.(f_jacobian[:, length(x̂ₖ)+length(uₖ)+1:end], digits = 10)

	h_jacobian = ForwardDiff.jacobian(h, BlockVector(vcat([x̂ₖ, n]...), [length(x̂ₖ), length(n)]))

	Hₖ = round.(h_jacobian[:, 1:length(x̂ₖ)], digits = 10)
	Nₖ = round.(h_jacobian[:, length(x̂ₖ)+1:end], digits = 10)

	Γₖ₊₁ = Aₖ * Σₖ * Aₖ' + Mₖ * Mₖ'
	# Main.@infiltrate any(isnan.(Hₖ * Γₖ₊₁ * Hₖ' + Nₖ * Nₖ'))
	Kₖ = Γₖ₊₁ * Hₖ' * (round.(Hₖ * Γₖ₊₁ * Hₖ' + Nₖ * Nₖ', digits = 100) \ I)

	noiseless_x̂ₖ₊₁ = env.state_dynamics(x̂ₖ, uₖ, m)

	temp  = env.observation_function(states = noiseless_x̂ₖ₊₁, m = n)
	x̂ₖ₊₁ = noiseless_x̂ₖ₊₁ + Kₖ * (observations - temp)
	Σₖ₊₁ = (I - Kₖ * Hₖ) * Γₖ₊₁

	x̂_temp = BlockVector(x̂ₖ₊₁, mean_lengths)
	Σ_block = BlockArray(Σₖ₊₁, cov_lengths, cov_lengths)
	temp = vcat([Σ_block[Block(ii, ii)] for ii in eachindex(players)]...)
	Σ_temp = BlockVector(vec(temp'), [cov_length^2 for _ in eachindex(players)])

	β_new = BlockVector(vcat([vcat(x̂_temp[Block(ii)], Σ_temp[Block(ii)]) for ii in eachindex(players)]...),
		[mean_lengths[i] + cov_lengths[i]^2 for i in eachindex(players)])
	return β_new, Aₖ, Mₖ, Hₖ, Nₖ, Kₖ, x̂_temp, Σ_block
end

end
