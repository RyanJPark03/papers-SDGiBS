# using BlockArrays

struct base_environment{}
	state_dynamics::Function
	observation_function::Function
	num_agents::Int
	state_dim::Int
	initial_state::BlockVector{Float64}
	current_state::BlockVector{Float64}
	time::Int
	final_time::Int
end


function init_base_environment(;
	state_dynamics::Function,
	observation_function::Function,
	num_agents::Int,
	state_dim::Int,
	initial_state::BlockVector{Float64},
	final_time::Int = -1)

	#TODO: assert that all dimensions line up correctly, potentially init some stuff

	base_environment(
		state_dynamics,
		observation_function,
		num_agents,
		state_dim,
		initial_state,
		initial_state,
		0,
		final_time)
end

function unroll(env::base_environment, actions::Vector{BlockVector{Float64}}, time_steps::Int)
	if time_steps + env.time > env.final_time
		println("Time steps exceed final time")
		return nothing
	end

	states::Array{Vector{Float64}} = [[] for _ in 1:time_steps+1]
	states[1] = env.current_state
	for tt in 2:time_steps+1
		states[tt] = env.state_dynamics(states[tt-1], actions)
	end
	env.time += time_steps
	env.current_state = states[end]
	return states[2:end]
end

function observations(env::base_environment)
	return env.observation_function(env.current_state)
end
