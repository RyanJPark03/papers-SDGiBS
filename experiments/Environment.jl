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

function unroll(env::base_environment, players;
    noise=true, noise_clip = true, noise_clip_val = .1, noise_scalar=1)
	if 1 + env.time > env.final_time
		println("Time steps exceed final time")
		return
	end

	actions = BlockVector{Float64}(undef, [player.action_space for player in players])
	cur_beliefs = vcat([player.belief for player in players]...)

	for ii in eachindex(players)
		actions[Block(ii)] .= get_action(players, ii, 1; state = cur_beliefs)
	end

	n = noise_scalar .* rand(Distributions.Normal(), env.dynamics_noise_dim*env.num_agents)
	if noise_clip
		n = min.(n, noise_clip_val)
		n = max.(n, -noise_clip_val)
	end
	dn = (noise) ? n : zeros(env.dynamics_noise_dim*env.num_agents)
	dyn_noise = BlockVector(dn, [env.dynamics_noise_dim for _ in 1:env.num_agents])
	env.current_state = env.state_dynamics(env.current_state, actions, dyn_noise)

	env.time += 1
end

function observations(env::base_environment)
	return env.observation_function(; states = env.current_state)
end
