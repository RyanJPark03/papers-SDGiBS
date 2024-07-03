module SDGiBS

using BlockArrays
using LinearAlgebra
using ForwardDiff
using ForwardDiff: Chunk, JacobianConfig, HessianConfig
using Distributions

export belief_update
function belief_update(env, players::Array, observations)
	return calculate_belief_variables(env, players, observations, env.time)[1]
end

export SDGiBS_solve_action
function SDGiBS_solve_action(players::Array, env) 
    Σₒ = BlockArray{Float64}(undef, [env.state_dim for player in players], [env.state_dim for player in players])
        for ii in eachindex(players)
            Σₒ[Block(ii, ii)] .= reshape(players[ii].history[env.time][2][env.state_dim + 1:end], (env.state_dim, env.state_dim))
        end
    
    # ū = BlockVector(vcat([player.predicted_control[env.time:end] for player in players]...),
    #                 [length(player.predicted_control[env.time:end]) for player in players])
    ū = BlockArray(hcat([vcat([player.predicted_control[tt] for player in players]...) for tt in env.time:env.final_time - 1]...),
                    [player.action_space for player in players], [1 for _ in env.time:env.final_time - 1])
    u_k = (tt) -> BlockVector(vcat(ū[Block(ii, tt)] for ii in eachindex(players)), [player.action_space for player in players])

    cₖ = (x) -> [player.cost(BlockVector(x[Block(1)], [env.state_dim for _ in players])
        , BlockVector(x[Block(2)], [p.action_space for p in players])) for player in players]
    cₗ = (x) -> [player.final_cost(x) for player in players]
    


    b̄, nominal_states = simulate(env, players, ū)[1]

    @assert length(b̄) == size(ū)[2] + 1
    @assert length(b̄) == env.final_time - env.time + 1

    # Iteration variables
    Q_old = [Inf for _ in eachindex(players)]
    ϵ = 1e-5
    Q_new = cost(players, b̄, ū)

    π = []
    c = [player.final_cost for player in players]

    while norm(Q_new - Q_old, 2) > ϵ
        # Bakcwards Pass
        # β, Aₖ, Mₖ, Hₖ, Nₖ, Kₖ, x̂ₖ₊₁, Σₖ₊₁ = calculate_belief_variables(env, players, observations, tt)

        V = cₗ(b̄[end])
        V_b = vcat(map((cᵢ) -> ForwardDiff.gradient(cᵢ, b̄[end]), c)...)
        # V_bb = ForwardDiff.hessian(c[1], b̄[end][1:end])
        V_bb = vcat(map((cᵢ) -> ForwardDiff.hessian(cᵢ, b̄[end][1:end]), c)...)
        error("holy moly it worked")
        for tt in env.final_time:-1:env.time
            # πₖ = ūₖ + jₖ + Kₖ * δbₖ
            # jₖ = -Q̂⁻¹ᵤᵤ * Q̂ᵤ
            # Kₖ = -Q̂⁻¹ᵤᵤ * Q̂ᵤ
            # TODO: update cₖ to match cₗ
            Qₖ = cₖ(b̄[tt], u_k(tt)) + V + .5 * sum()
        end
    end

    error("skipped while loop")
	return [1, 1]
end

function simulate(env, players, ū; noise = false)
    b̄ = [BlockVector{Float64}(undef, [env.state_dim + env.state_dim^2 for _ in eachindex(players)]) for _ in 1:env.final_time - env.time + 1]
    sts = [BlockVector{Float64}(undef, [env.state_dim for _ in eachindex(players)]) for _ in 1:env.final_time - env.time + 1]

    belief_length = length(players[1].history[end][2])
    b̄[1] = BlockVector(vcat([player.history[env.time][2] for player in players]...),
        [belief_length for player in players])
    sts[1] .= env.current_state

    dynamics_noise = BlockVector(zeros(env.dynamics_noise_dim * length(players)), [env.dynamics_noise_dim for _ in 1:length(players)])
    observation_noise = BlockVector(zeros(env.observation_noise_dim * length(players)), [env.observation_noise_dim for _ in 1:length(players)])
  
    for tt in 1:env.final_time - env.time
        # belief_st = get_prefixes(b̄[tt], env.state_dim)
        actual_time = tt + env.time - 1
        action = BlockVector(vec(vcat([ū[Block(i, actual_time)] for i in eachindex(players)]...)),
            [player.action_space for player in players])
        
        if noise
            dynamics_noise .= [rand(Distributions.Normal()) for _ in 1 : env.dynamics_noise_dim * length(players)]
            observation_noise .= [rand(Distributions.Normal()) for _ in 1 : env.observation_noise_dim * length(players)]
        end
        sts[tt + 1] .= env.state_dynamics(sts[tt], action, dynamics_noise)
        observations = env.observation_function(states = sts[tt + 1], m = observation_noise)

        β, Aₖ, Mₖ, Hₖ, Nₖ, Kₖ, x̂ₖ₊₁, Σₖ₊₁ = calculate_belief_variables(env, players, observations, actual_time)
        temp = []
        for ii in eachindex(players)
            temp = vcat(temp, vcat(x̂ₖ₊₁[Block(ii)], vec(Σₖ₊₁[Block(ii, ii)])))
        end
        b̄[tt + 1] .= temp
    end

    return b̄, sts
end

function cost(players, b, u)
    cₖ = [player.cost for player in players]
    cₗ = [player.final_cost for player in players]

    Q = [0.0 for _ in eachindex(players)]

    for ii in eachindex(players)
        for tt in 1 : length(blocks(b)) - 1
            actions = BlockVector(vcat([u[Block(ii, tt)] for ii in eachindex(players)]...),
                [player.action_space for player in players])
            Q[ii] += cₖ[ii](b[tt], actions)
        end
        Q[ii] += cₗ[ii](b[end])
    end
    
    return Q
end


function get_prefixes(b̄, prefix_length::Int)
    return BlockVector(vcat([b̄[Block(ii)][1:prefix_length] for ii in eachindex(blocks(b̄))]...),
        [prefix_length for _ in eachindex(blocks(b̄))])
end

function calculate_belief_variables(env, players, observations, time)
    num_players = length(players) 
    # TODO: this is at env.time, not at input time
    β = BlockVector(vcat([player.belief for player in players]...), [length(player.belief) for player in players])


    mean_lengths = [env.state_dim for _ in 1:num_players]
    cov_length = Int(sqrt(Int(length(β) // num_players - env.state_dim)))
    cov_lengths = [cov_length for _ in 1:num_players]

    x̂ₖ = BlockVector(vcat([β[Block(ii)][1:env.state_dim] for ii in 1:num_players]...), mean_lengths)
    Σₖ = BlockArray{Float64}(undef, cov_lengths, cov_lengths)

    for ii in eachindex(players)
        for jj in eachindex(players)
            if ii == jj
                Σₖ[Block(ii, jj)] .= reshape(β[Block(ii)][env.state_dim+1:end], tuple(cov_lengths...))
            else
                Σₖ[Block(ii, jj)] .= zeros((cov_length, cov_length))
            end
        end
    end

    # TODO: not entirely sure what design choice I want with env.time + 1 here (depending on how env.time is init)
    uₖ = BlockVector(vcat([player.predicted_control[time] for player in players]...), 
                    [player.action_space for player in players])
    m = BlockVector([0.0 for _ in 1:env.dynamics_noise_dim * num_players],
                    [env.dynamics_noise_dim for _ in 1:num_players])

    f = (x) -> env.state_dynamics(
        BlockVector(x[Block(1)], mean_lengths),
        BlockVector(x[Block(2)], [player.action_space for player in players]),
        BlockArray(x[Block(3)], [env.dynamics_noise_dim for _ in 1:num_players]), block=false)
        
    h = (x) -> env.observation_function(
        states = BlockVector(x[Block(1)], mean_lengths),
        m = BlockVector(x[Block(2)], [env.observation_noise_dim for _ in 1:num_players]), block = false)

    j_cfg = JacobianConfig(f, BlockVector(vcat([x̂ₖ, uₖ, m]...), [length(x̂ₖ), length(uₖ), length(m)]), Chunk{20}())
    f_jacobian = ForwardDiff.jacobian(f, BlockVector(vcat([x̂ₖ, uₖ, m]...), [length(x̂ₖ), length(uₖ), length(m)]), j_cfg)
    Aₖ = f_jacobian[:, 1:length(x̂ₖ)]
    Mₖ = f_jacobian[:, length(x̂ₖ) + length(uₖ)+ 1:end]

    n = BlockVector([0.0 for _ in 1:env.observation_noise_dim * num_players],
                    [env.observation_noise_dim for _ in 1:num_players])

    # Main.@infiltrate

    h_jacobian = ForwardDiff.jacobian(h, BlockVector(vcat([x̂ₖ, n]...), [length(x̂ₖ), length(n)]))

    Hₖ = h_jacobian[:, 1:length(x̂ₖ)]
    Nₖ = h_jacobian[:, length(x̂ₖ) + 1:end]

    # Main.@infiltrate

    Γₖ₊₁ = Float64.(Aₖ) * Σₖ * Float64.(Aₖ') + Mₖ * Mₖ'
    Kₖ = Γₖ₊₁ * Hₖ' * ((Float64.(Hₖ) * Γₖ₊₁ * Float64.(Hₖ') + Float64.(Nₖ) * Float64.(Nₖ')) \ I)


    noiseless_x̂ₖ₊₁ = env.state_dynamics(x̂ₖ, uₖ, m)
    zeroed_env_noise = BlockVector([0.0 for _ in 1:env.observation_noise_dim * num_players], [env.observation_noise_dim for _ in eachindex(players)])
    
    temp  = env.observation_function(states = noiseless_x̂ₖ₊₁, m = zeroed_env_noise)
    # Main.@infiltrate
    x̂ₖ₊₁ = noiseless_x̂ₖ₊₁ + Float64.(Kₖ) * (observations - temp)
    Σₖ₊₁ = (I - Kₖ * Hₖ) * Γₖ₊₁

    x̂_temp = BlockVector(x̂ₖ₊₁, mean_lengths)
    Σ_block = BlockArray(Σₖ₊₁, cov_lengths, cov_lengths)
    temp = vcat([Σ_block[Block(ii, ii)] for ii in eachindex(players)]...)
    Σ_temp = BlockVector(vec(temp), [cov_length^2 for _ in eachindex(players)])

    β = BlockVector(vcat([vcat(x̂_temp[Block(ii)], Σ_temp[Block(ii)]) for ii in eachindex(players)]...), 
                    [mean_lengths[i] + cov_lengths[i]^2 for i in eachindex(players)])
    return β, Aₖ, Mₖ, Hₖ, Nₖ, Kₖ, x̂_temp, Σ_block
end

end
