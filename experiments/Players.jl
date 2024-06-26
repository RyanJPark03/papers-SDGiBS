# using BlockArrays

# include("Environment.jl")

export player_type
@enum player_type begin
	no_change
	random
	SDGiBS
end

export player
struct player{}
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
	# SDGiBS specific
	predicted_belief::Vector{Vector{Float64}}
	predicted_control::Vector{Vector{Float64}}
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
	default_action::Vector{Float64} = nothing,
	time::Int = 1)

	# Check inputs are good
	if Integer(player_type) < 0 || isnothing(belief) || action_space < 0
		error("One or more inputs must be specified")
	end

	# everyone should use the same belief updater (from the paper)
	# belief_updater :: Function = () -> true
    
    # probably vestigial since everyone should use the same belief updater
	belief_updater::Function = (player::player, observation::Vector{Float64}) ->
		belief_update(player.belief, observation)
	action_selector::Function = () -> true

	predicted_belief = [belief for _ in 1:time+1]
	predicted_control = [zeros(action_space) for _ in 1:time+1]
	feedback_law = nothing

	if player_type == no_change
		action_selector = (player::player, observation::Vector{Float64}) -> default_action
	elseif player_type == random
		action_selector = (player::player, observation::Vector{Float64}) ->
			rand(action_space)
	elseif player_type == SDGiBS
		# TODO: some optimization if all players are SDGiBS
		action_selector = handle_SDGiBS_action

	else
		error("Unimplemented player type or unknown player type $player_type")
		return nothing
	end

	player(player_type, player_id, belief, cost, final_cost, [], belief_updater, action_selector, action_space,
		predicted_belief, predicted_control, feedback_law)
end

function handle_SDGiBS_action(players::Array{player}, env::base_environment,
	current_player_index::Int)
    # probably no π
	(b̄, ū, π) = SDGiBS_solve_action(players, env)
	players[current_player_index].predicted_belief = b̄[Block(current_player_index)]
	players[current_player_index].predicted_control = ū[Block(current_player_index)]
	players[current_player_index].feedback_law = π
    return players[current_player_index].predicted_control[env.time]
end

# Vestigial, using time_step_all
function time_step(player_index::Int = -1, observation::Vector{Float64},
	env::base_environment)
	# TODO: What is the order? Observation -> update belief -> select action

	player = env.players[player_index]

	# Record observation
	push!(player.history, [observation])

	# Players start at "time 0" with a prior belief
	# TODO: start environment at time 1, initilize priors
	@assert length(player.history) == env.time

	# Update belief and record
	player.belief = player.belief_updater(player, observation)
	push!(player.history[end], player.belief)

	# Select action
	if player.player_type == SDGiBS
		return player.action_selector(env.players, env, player_index)
	else
		return player.action_selector(player, observation)
	end
end

export time_step_all
function time_step_all(players::Array{player}, env::base_environment, observations::BlockVector{Float64})
    # Observation, belief, then action, lets say we start with prior -> observation -> belief ->action
    # We count the first action as time 1

    # update all beliefs
    # temporarily store all new beliefs

    # Got observations already

    # Get updated beliefs
    new_beliefs = belief_update(env, players, observations)

    # Do actions
    all_actions = BlockVector{Float64}(undef, [player.action_space for player in players])
    for ii in eachindex(players)
        player = players[ii]
        push!(player.history, [observations[Block(ii)], player.belief])
        player.belief = new_beliefs[Block(ii)]

        if player.player_type == SDGiBS
            all_actions[Block(ii)] .= handle_SDGiBS_action(players, env, ii)
        else
            all_actions[Block(ii)] .= action_selector(player, observations[Block(ii)])
        end
        push!(player.history[end], all_actions[Block(ii)])
    end

    return all_actions
end