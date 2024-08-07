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
	history::Array{Any}
	solver_history::Array{Any}
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
		final_time,
		[initial_state],
		[[]])
end

function unroll(env::base_environment, players;
    noise=false, noise_clip = false, noise_clip_val = .1, store_solver_history = false)
	println("unrolling")
	if 1 + env.time > env.final_time
		println("Time steps exceed final time")
		return
	end

	actions = BlockVector{Float64}(undef, [player.action_space for player in players])
	cur_beliefs = vcat([player.belief for player in players]...)

	for ii in eachindex(players)
		act = get_action(players, ii, 1; state = cur_beliefs)
		println("action for player: ", ii, ": ", act)
		actions[Block(ii)] .= act
	end

	n = rand(Distributions.Normal(0.0, 1.0), env.dynamics_noise_dim*env.num_agents)
	if noise_clip
		n = min.(n, noise_clip_val)
		n = max.(n, -noise_clip_val)
	end
	dn = (noise) ? n : zeros(env.dynamics_noise_dim*env.num_agents)
	dyn_noise = BlockVector(dn, [env.dynamics_noise_dim for _ in 1:env.num_agents])
	env.current_state = env.state_dynamics(env.current_state, actions, dyn_noise)
	push!(env.history, env.current_state)
	if store_solver_history
		push!(env.solver_history, [])
	end
	println("new state: ", env.current_state)

	env.time += 1
end

function observations(env::base_environment)
	return env.observation_function(; states = env.current_state)
end


function num_agents(env::base_environment)
	return env.num_agents
end

function state_dim(env::base_environment)
	return env.state_dim
end