using SDGiBS
using Distributions

export player_type
@enum player_type begin
	no_change
	random
	type_SDGiBS
end

export Player
mutable struct Player{}
	player_type::player_type
	player_id::Int
	belief::Vector{Float64}
	cost::Function
	final_cost::Function
	# tuple of (observation, belief, action) pairs
	history::Array
	belief_updater::Function
	action_selector::Function
	action_space::Int
	observation_space::Int
	# SDGiBS specific
	predicted_belief::Vector{Any}
	predicted_control::Vector{Any}
	feedback_law::Any
	# solver_iter_work::Array{Any}
end

export init_player
function init_player(;
	player_type::player_type = -1,
	player_id::Int = -1,
	belief = nothing,
	cost::Function,
	final_cost::Function,
	action_space::Int = -1,
	observation_space::Int = -1,
	default_action::Vector{Float64} = nothing,
	time::Int = 1,
	num_players::Int = 1)

	# Check inputs are good
	if Integer(player_type) < 0 || isnothing(belief) || action_space < 0
		error("One or more inputs must be specified")
	end

	# probably vestigial since everyone should use the same belief updater
	belief_updater::Function = (player::Player, observation::Vector{Float64}) ->
		belief_update(player.belief, observation)
	action_selector::Function = () -> true
	predicted_belief = [copy(belief) for _ in 1:time+1]
	predicted_control = [BlockVector(vcat([copy(default_action) for _ in 1:num_players]...), [length(default_action), length(default_action)]) for _ in 1:time+1]
	feedback_law = nothing
	# Main.@infiltrate

	if player_type == no_change
		action_selector = (player::Player, observation::Vector{Float64}) -> default_action
	elseif player_type == random
		action_selector = (player::Player, observation::Vector{Float64}) ->
			rand(action_space)
	elseif player_type == type_SDGiBS
		action_selector = (players, ii, time, state) -> get_action(players, ii, time; state)

	else
		error("Unimplemented player type or unknown player type $player_type")
		return nothing
	end

	Player(player_type, player_id, copy(belief), cost, final_cost,
		[[copy(default_action), nothing, copy(belief)]], belief_updater, action_selector,
		action_space, observation_space, predicted_belief, predicted_control,
		feedback_law)
end

function handle_SDGiBS_action(players::Array{Player}, env::base_environment,
	current_player_index::Int, action_selector, time::Int = 1)
	(b̄, ū, π) = SDGiBS_solve_action(players, env, action_selector)
	players[current_player_index].predicted_belief = b̄
	players[current_player_index].predicted_control = ū
	players[current_player_index].feedback_law = π
end

function handle_SDGiBS_action_coop(players::Array{Player}, env::base_environment, action_selector, time::Int = 1)
	(b̄, ū, π) = SDGiBS_solve_action(players, env, action_selector; μᵦₒ = 1.0, μᵤₒ = 1.0, horizon = 10)
	for ii in eachindex(players)
		players[ii].predicted_belief = b̄
		players[ii].predicted_control = ū
		players[ii].feedback_law = [(δb) -> BlockVector(π[jj](δb), [player.action_space for player in players])[Block(ii)] for jj in eachindex(π)]
	end
end


function get_nominal_belief(current_player, time)
	return current_player.predicted_belief[time]
end

function get_δb(current_player, time, state)
	println("player: ", current_player.player_id, " time: ", time, " norm δb: ", norm(state - get_nominal_belief(current_player, time)))
	return state - get_nominal_belief(current_player, time)
end

"""
gets a predicted action for a player, given the current state

params: players - array of players
		ii - index of the player
		steps_ahead - how many steps ahead to look
			This one is a little confusing, but a player's predicted control starts at the current time step, so steps_ahead = 1 returns the predicted control
			at the current time step. steps_ahead = 2 returns the predicted control at the next time step, etc...
		state - the current state of the world
"""

export get_action
function get_action(players, ii, steps_ahead; state=nothing)
	if isnothing(players[ii].feedback_law) || isnothing(state) 
		return players[ii].history[end][1]
	else
		if steps_ahead > length(players[ii].feedback_law)
			return zeros(size(players[ii].history[end][1]))
		end
		return players[ii].feedback_law[steps_ahead](get_δb(players[ii], steps_ahead, state))
	end
end

export time_step_all
function time_step_all(players::Array{Player}, env::base_environment)
	# println("time step: ", env.time, " / ", env.final_time)
	# Act, Obs, Upd

	# Act
	for ii in eachindex(players)
		player = players[ii]
		if env.time == env.final_time
			player.predicted_control = zeros(player.action_space)
		elseif player.player_type == type_SDGiBS
			handle_SDGiBS_action(players, env, ii, get_action, env.time)
		else
			action_selector(player, observations[Block(ii)])
		end
	end

	# Iterate environment
	unroll(env, players; noise = false)

	# Do observations
	motion_noise = 1.0
	m = BlockVector(vcat([rand(-motion_noise:motion_noise, env.observation_noise_dim) for _ in 1:env.num_agents]...),
	[env.observation_noise_dim for _ in 1:env.num_agents])

	observations = env.observation_function(; states = BlockVector(env.current_state, [4, 4]), m = m)

	# Get updated beliefs
	old_beliefs = vcat([player.belief for player in players]...)
	# new_beliefs = [SDGiBS.belief_update(env, players, observations)[Block(ii)] for ii in 1:env.num_agents]
	new_beliefs = SDGiBS.belief_update(env, players, observations)
	
	for ii in eachindex(players)
		player = players[ii]
		player.belief = new_beliefs[Block(ii)]
		println("grabbing action, player: ", ii, " time: ", env.time)
		push!(player.history, [get_action(players, ii, 1; state = old_beliefs), observations[Block(ii)], player.belief])
	end
end

export time_step_all_coop
function time_step_all_coop(players::Array{Player}, env::base_environment)
	# Act, Obs, Upd

	# Act
	handle_SDGiBS_action_coop(players, env, get_action, env.time)
	cur_beliefs = get_full_belief_state(players)
	actions = BlockVector(
		vcat([get_action(players, ii, 1; state = cur_beliefs) for ii in eachindex(players)]...),
		[player.action_space for player in players])
	
	# Iterate environment
	unroll(env, players; noise = false, noise_clip = false, noise_clip_val = 0.1)

	m = BlockVector(vcat([rand(Distributions.Normal(0.0, 1.0), env.observation_noise_dim) for _ in 1:env.num_agents]...),
	[env.observation_noise_dim for _ in 1:env.num_agents])
	observations = env.observation_function(; states = BlockVector(env.current_state, [4, 4]), m = m)
	println("actual observations: ", observations)
	println("actual state: ", env.current_state)
	println("diff: ", norm(observations - env.current_state))
	# Get updated beliefs
	old_beliefs = get_full_belief_state(players)
	# new_beliefs = [SDGiBS.belief_update(env, players, observations)[Block(ii)] for ii in 1:env.num_agents]
	new_beliefs = SDGiBS.belief_update(env, players, observations, actions)

	obs_spaces = [players[ii].observation_space for ii in eachindex(players)]
	
	for ii in eachindex(players)
		player = players[ii]
		player.belief = new_beliefs[Block(ii)]
		action = get_action(players, ii, 1; state = old_beliefs)
		# Main.@infiltrate
		push!(player.history, [action, observations[sum(obs_spaces[1:ii-1])+1:sum(obs_spaces[1:ii])], player.belief])
	end
end

export get_history
function get_history(player::Player, time::Int)
	if isnothing(time)
		return player.history
	end
	return player.history[time]
end

export get_full_belief_state
function get_full_belief_state(players::Array{Player})
	return vcat([player.belief for player in players]...)
end

function get_belief_mean(player; block = false)
	if block
		return BlockVector(player.belief[1:player.observation_space], [player.observation_space])
	else
		return player.belief[1:player.observation_space]
	end
end

function get_belief_cov(player; block = false)
	if block
		return BlockMatrix(player.belief[player.observation_space+1:end], [player.observation_space, player.observation_space])
	else
		return player.belief[player.observation_space+1:end]
	end
end
	
function get_observation_space(player::Player)
	return player.observation_space
end

function get_action_space(player::Player)
	return player.action_space
end