# using BlockArrays

mutable struct base_environment{}
	state_dynamics::Function
	observation_function::Function
	num_agents::Int
	state_dim::Int
    dynamics_noise_dim::Int
    observation_noise_dim::Int
	initial_state
	current_state
	time::Int
	final_time::Int
end


function init_base_environment(;
	state_dynamics::Function,
	observation_function::Function,
	num_agents::Int,
	state_dim::Int,
    dynamics_noise_dim::Int,
    observation_noise_dim::Int,
	initial_state,
	final_time::Int = -1)

	base_environment(
		state_dynamics,
		observation_function,
		num_agents,
		state_dim,
        dynamics_noise_dim,
        observation_noise_dim,
		initial_state,
		initial_state,
		1,
		final_time)
end

function unroll(env::base_environment, actions, time_steps::Int;
    noise::Vector{Float64}=nothing)
	if time_steps + env.time > env.final_time
		println("Time steps exceed final time")
		return nothing
	end

	states::Array{Vector{Float64}} = [[] for _ in 1:time_steps+1]
	states[1] = env.current_state
	for tt in 2:time_steps+1
        if !isnothing(noise)
            states[tt] = env.state_dynamics(states[tt-1], actions, noise)
        else
            states[tt] = env.state_dynamics(states[tt-1], actions)
        end
	end
	env.time += time_steps
	env.current_state = states[end]
	if time_steps == 1
		return states[2]
	end
	return states[2:end]
end

function observations(env::base_environment)
	return env.observation_function(; states = env.current_state)
end
