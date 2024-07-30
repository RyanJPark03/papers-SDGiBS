mutable struct SDG_Player <: AbstractPlayer
    player_id::Int,
    cost::Function,
    intermediate_cost::Function,
    final_cost::Function,
    action_space::Int,
    self_observation_space::Int,
    self_observation_function::Function,
    other_observation_spaces::Tuple{Int, Function},
    other_observation_function::Tuple{Int, Function}
end