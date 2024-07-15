# using BlockArrays

using SDGiBS

export player_type
@enum player_type begin
	no_change
	random
	type_SDGiBS
end

export player
mutable struct player{}
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
	belief_updater::Function = (player::player, observation::Vector{Float64}) ->
		belief_update(player.belief, observation)
	action_selector::Function = () -> true
	#TODO: initalizing on top of eachother will  cause evasive manuvers at first
	predicted_belief = [BlockVector(vcat([copy(belief) for _ in 1:num_players]...), [length(belief), length(belief)]) for _ in 1:time+1]
	predicted_control = [BlockVector(vcat([copy(default_action) for _ in 1:num_players]...), [length(default_action), length(default_action)]) for _ in 1:time+1]
	feedback_law = nothing

	if player_type == no_change
		action_selector = (player::player, observation::Vector{Float64}) -> default_action
	elseif player_type == random
		action_selector = (player::player, observation::Vector{Float64}) ->
			rand(action_space)
	elseif player_type == type_SDGiBS
		action_selector = handle_SDGiBS_action

	else
		error("Unimplemented player type or unknown player type $player_type")
		return nothing
	end

	player(player_type, player_id, copy(belief), cost, final_cost,
    [[nothing, copy(belief), default_action]], belief_updater, action_selector, # TODO: history should be empty when player initialized
    action_space, observation_space, predicted_belief, predicted_control,
    feedback_law)
end

function handle_SDGiBS_action(players::Array{player}, env::base_environment,
	current_player_index::Int, time::Int = 1)
    # probably no π
	(b̄, ū, π) = SDGiBS_solve_action(players, env, time)
	players[current_player_index].predicted_belief = b̄
	players[current_player_index].predicted_control = ū
	players[current_player_index].feedback_law = π
    return players[current_player_index].predicted_control[1]
end

export time_step_all
function time_step_all(players::Array{player}, env::base_environment, observations)
	println("time stepping at: ", env.time, " / ", env.final_time)
    # Observation, belief, then action, lets say we start with prior -> observation -> belief ->action
    # We count the first action as time 1

    # update all beliefs
    # temporarily store all new beliefs

    # Got observations already#TODO: order wrong probably

    # Do actions
    all_actions = BlockVector{Float64}(undef, [player.action_space for player in players])
    # total_action_space = sum([player.action_space for player in players])
    # all_actions = [0.0 for _ in 1 : total_action_space]
    for ii in eachindex(players)
        player = players[ii]
        push!(player.history, [observations[Block(ii)], player.belief])
	end

	# Get updated beliefs
    new_beliefs = SDGiBS.belief_update(env, players, observations)
	println("new_beliefs: ", new_beliefs)

	for ii in eachindex(players)
		player = players[ii]
        # push!(player.history, [observations[player.observation_space * (ii - 1) + 1 : player.observation_space * ii], player.belief])
        
        # total_prev_belief_space = sum([player.observation_space for player in players[1:ii]])
        # player.belief = new_beliefs[total_prev_belief_space + 1 : total_prev_belief_space + player.observation_space]

        # total_prev_action_space = sum([player.action_space for player in players[1:ii]])
		if env.time == env.final_time
			# Main.@infiltrate
			all_actions[Block(ii)] .= zeros(player.action_space)
		elseif player.player_type == type_SDGiBS
			temp = handle_SDGiBS_action(players, env, ii, env.time)
			# Main.@infiltrate
            all_actions[Block(ii)] .= temp[Block(ii)]
            # all_actions[total_prev_action_space + 1 : total_prev_action_space + player.action_space] = handle_SDGiBS_action(players, env, ii)
            # all_actions
        else
            all_actions[Block(ii)] .= action_selector(player, observations[Block(ii)])
            # all_actions[total_prev_action_space + 1 : total_prev_action_space + player.action_space] = player.action_selector(player, observations[total_prev_belief_space + 1 : total_prev_belief_space + player.observation_space])
        end
        push!(player.history[end], all_actions[Block(ii)])
        # push!(player.history[end], all_actions[total_prev_action_space + 1 : total_prev_action_space + player.action_space])
    end

	for ii in eachindex(players)	
		player = players[ii]
		player.belief .= new_beliefs[Block(ii)]
	end

    return all_actions
end