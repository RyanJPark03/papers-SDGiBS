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
    player_type :: player_type
    player_id :: Int
    belief :: Vector{Float64}
    cost :: Function
    final_cost :: Function
    # tuple of (belief, observation, action) pairs
    history :: Array{Tuple{BlockVector{Float64}, BlockVector{Float64}, BlockVector{Float64}}}
    belief_updater :: Function
    action_selector :: Function
    action_space :: Int
    # SDGiBS specific
    predicted_belief :: BlockVector{Float64}
    predicted_control :: BlockVector{Float64}
    feedback_law :: Function
end

export init_player
function init_player(;
    player_type :: player_type = -1,
    player_id :: Int = -1,
    belief :: Vector{Float64} = nothing,
    cost :: Function,
    final_cost :: Function,
    action_space :: Int = -1,
    default_action :: Vector{Float64} = nothing,
    time :: Int = 1)

    # Check inputs are good
    player_type < 0 || isnothing(belief) || action_space < 0 || error("One or more inputs must be specified")

    # everyone should use the same belief updater (from the paper)
    # belief_updater :: Function = () -> true
    belief_updater :: Function = (player :: player, observation :: Vector{Float64}) ->
        belief_update(player.belief, observation)
    action_selector :: Function = () -> true

    predicted_belief = [belief for _ in 1:time + 1]
    predicted_control = [zeros(action_space) for _ in 1:time + 1]
    feedback_law = nothing

    if player_type == no_change
        action_selector = (player :: player, observation :: Vector{Float64}) -> default_action
    elseif player_type == random
        action_selector = (player :: player, observation :: Vector{Float64}) ->
            rand(action_space)
    elseif player_type == SDGiBS
        # TODO: some optimization if all players are SDGiBS
        action_selector = handle_SGDiBS_action
            
    else
        error("Unimplemented player type or unknown player type $player_type")
        return nothing
    end

    player(player_type, player_id, belief, cost, final_cost, [], belief_updater, action_selector, action_space,
        predicted_belief, predicted_control, feedback_law)
end

function handle_SBDiBS_action(players :: Array{player}, env :: base_environment, current_player_index :: Int)
    (b̄, ū, π) = SDGiBS_solve_action(players, env, current_player_index)
    players[current_player_index].predicted_belief = b̄[Block(current_player_index)]
    players[current_player_index].predicted_control = ū[Block(current_player_index)]
    players[current_player_index].feedback_law = π
end
