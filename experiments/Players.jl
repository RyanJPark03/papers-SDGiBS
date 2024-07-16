using SDGiBS

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
end

export init_player
function init_player(;
	player_type::player_type = -1,
	player_id::Int = -1,
	belief::Vector{Float64} = nothing,
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
	#TODO: initalizing on top of eachother will  cause evasive manuvers at first
	predicted_belief = [BlockVector(vcat([copy(belief) for _ in 1:num_players]...), [length(belief), length(belief)]) for _ in 1:time+1]
	predicted_control = [BlockVector(vcat([copy(default_action) for _ in 1:num_players]...), [length(default_action), length(default_action)]) for _ in 1:time+1]
	feedback_law = nothing

	if player_type == no_change
		action_selector = (player::Player, observation::Vector{Float64}) -> default_action
	elseif player_type == random
		action_selector = (player::Player, observation::Vector{Float64}) ->
			rand(action_space)
	elseif player_type == type_SDGiBS
		action_selector = (players, ii, time, state) -> get_action(players, ii, time, state)

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
	# probably no π
	(b̄, ū, π) = SDGiBS_solve_action(players, env, action_selector)
	players[current_player_index].predicted_belief = b̄
	players[current_player_index].predicted_control = ū
	players[current_player_index].feedback_law = π
	# return players[current_player_index].predicted_control[1]
end

function get_nominal_belief(current_player, time)
	return current_player.predicted_belief[time]
end

function get_δb(current_player, time, state)
	return state - get_nominal_belief(current_player, time)
end

export get_action
function get_action(players, ii, time; state=nothing, nominal_state = nothing, δb = nothing)
	if isnothing(players[ii].feedback_law)
		return players[ii].history[end][1]
	elseif !isnothing(state) && isnothing(nominal_state) && isnothing(δb)
		return players[ii].feedback_law[time](get_δb(players[ii], time, state))[sum([players[jj].action_space for jj in 1:ii-1])+1:sum([players[jj].action_space for jj in 1:ii])]
	elseif !isnothing(state) && !isnothing(nominal_state) && isnothing(δb)
		return players[ii].feedback_law(state - nominal_state)[sum([players[jj].action_space for jj in 1:ii-1])+1:sum([players[jj].action_space for jj in 1:ii])]
	else # state and nominal state are nothing, δb is not
		return players[ii].feedback_law(δb)[sum([players[jj].action_space for jj in 1:ii-1])+1:sum([players[jj].action_space for jj in 1:ii])]
	end
end

export time_step_all
function time_step_all(players::Array{Player}, env::base_environment)
	println("time step: ", env.time, " / ", env.final_time)
	# Act, Obs, Upd

	# Act
	for ii in eachindex(players)
		player = players[ii]
		if env.time == env.final_time
			# all_actions[Block(ii)] .= zeros(player.action_space)
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
	m = BlockVector(vcat([rand(-motion_noise:motion_noise, demo.env.observation_noise_dim) for _ in 1:demo.env.num_agents]...),
	[demo.env.observation_noise_dim for _ in 1:demo.env.num_agents])
	# m = BlockVector(zeros(sum([env.observation_noise_dim for _ in 1:env.num_agents])), [env.observation_noise_dim for _ in 1:env.num_agents])

	observations = env.observation_function(; states = BlockVector(env.current_state, [4, 4]), m = m)

	# Get updated beliefs
	old_beliefs = vcat([player.belief for player in players]...)
	new_beliefs = SDGiBS.belief_update(env, players, observations)
	for ii in eachindex(players)
		player = players[ii]
		player.belief .= new_beliefs[Block(ii)]
		push!(player.history, [get_action(players, ii, 1; state = old_beliefs), observations[Block(ii)], player.belief])
	end
end