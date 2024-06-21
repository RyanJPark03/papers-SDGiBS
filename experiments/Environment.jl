struct Base_Environment {}
    state_dynamics :: Function
    belief_dynamics :: Function
    observation_function :: Function
    num_agents :: Int
    state_dim :: Int
    initial_state :: Vector{Float64}
    current_state :: Vector{Float64}
    time :: Int
    final_time :: Int
end


function Base_Environment(;
    state_dynamics :: Function,
    belief_dynamics :: Function,
    observation_function :: Function,
    num_agents :: Int,
    state_dim :: Int,
    initial_state :: Vector{Float64},
    current_state :: Vector{Float64},
    time :: Int,
    final_time :: Int = -1)

    #TODO: assert that all dimensions line up correctly, potentially init some stuff

    Base_Environment{}(
    state_dynamics,
    belief_dynamics,
    observation_function,
    num_agents,
    state_dim,
    initial_state,
    current_state,
    time,
    final_time = final_time)
end

function unroll(env :: Base_Environment, actions :: Vector{Vector{Float64}}, time_steps :: Int)
    states :: Array{Vector{Float64}} = [[] for _ in 1 : time_steps + 1]
    states[1] = env.current_state
    for tt in 2 : time_steps + 1
        states[tt] = env.state_dynamics(states[tt - 1], actions)
    end
    return states
end

function observations(env :: Base_Environment)
    return env.observation_function(env.current_state)
end