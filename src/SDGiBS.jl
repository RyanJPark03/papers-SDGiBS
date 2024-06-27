module SDGiBS

using BlockArrays
using Symbolics

export belief_update
function belief_update(env, players::Array, observations::BlockVector{Float64})
    println("size of belief: ", size(players[1].belief))
    β = BlockVector(vcat([player.belief for player in players]...), [length(player.belief) for player in players])

    mean_lengths = [env.state_dim for _ in eachindex(players)]
    cov_lengths = [Int(sqrt(length(β[Block(1)]) - env.state_dim)) for _ in eachindex(players)]

    x̂ₖ = BlockVector(vcat([β[Block(ii)][1:env.state_dim] for ii in eachindex(players)]...), mean_lengths)
    Σₖ = BlockArray(vcat([reshape(β[Block(ii)][env.state_dim+1:end],
        tuple(cov_lengths...)) for ii in eachindex(players)]...), cov_lengths, [cov_lengths[1]])
    # TODO: not entirely sure what design choice I want with env.time + 1 here (depending on how env.time is init)
    uₖ = BlockVector(vcat([player.predicted_control[env.time+1] for player in players]...), 
                    [player.action_space for player in players])

    # TODO: for some reason m̃[1:env.noise_dim] is not working, env.noise_dim not found...
    @variables x̃[1:env.state_dim], ũ[1:sum([player.action_space for player in players])]
    @variables ñ[1:env.observation_noise_dim], m̃[1:env.dynamics_noise_dim]
    # @variables x̃[1:env.state_dim], ũ[1:sum([player.action_space for player in players])], m̃[1:2]
    x = Symbolics.scalarize(x̃)
    u = Symbolics.scalarize(ũ)
    m = Symbolics.scalarize(m̃)
    n = Symbolics.scalarize(ñ)

    f = (x, u, m) -> env.state_dynamics(x, u, m)
    h = (x, n) -> env.observation_function(;state=x, noise=n)

    x̂ₖ₊₁ = env.state_dynamics(x̂ₖ, uₖ, zeros(env.dynamics_noise_dim))

    # TODO: noise should be normally distributed
    A = Symbolics.build_function(Symbolics.gradient(f, x), expression = Val{false})[1]
    M = Symbolics.build_function(Symbolics.gradient(f, m), expression = Val{false})[1]
    H = Symbolics.build_function(Symbolics.gradient(h, x), expression = Val{false})[1]
    N = Symbolics.build_function(Symbolics.gradient(h, n), expression = Val{false})[1]
    
    Aₖ = A(x̂ₖ, uₖ, zeros(env.dynamics_noise_dim))
    Mₖ = M(x̂ₖ, uₖ, zeros(env.dynamics_noise_dim))
    Hₖ = H(x̂ₖ₊₁, zeros(env.observation_noise_dim))
    Nₖ = N(x̂ₖ₊₁, zeros(env.observation_noise_dim))

    println(typeof(Aₖ))
    println(size(Aₖ))
    println(Aₖ)
    println(typeof(Σₖ))
    println(size(Σₖ))
    println(Σₖ)

    Γₖ₊₁ = Aₖ * Σₖ * Aₖ' + Mₖ * Mₖ'
    Kₖ = Γₖ₊₁ * Hₖ' * ((Hₖ * Γₖ₊₁ * Hₖ' + Nₖ * Nₖ') \ I)

    x̂ₖ₊₁ = x̂ₖ₊₁ + Kₖ * (observations - h(x̂ₖ₊₁, [0, 0]))
    Σₖ₊₁ = (I - Kₖ * Hₖ) * Γₖ₊₁

    x̂_temp = BlockVector(x̂ₖ₊₁, state_lengths)
    Σ_temp = BlockVector(Σₖ₊₁, cov_lengths)

    β = BlockVector(vcat([vcat(x̂_temp[Block(ii)], Σ_temp[Block(ii)]) for ii in eachindex(env.players)]...), 
                    [state_lengths[i] + cov_lengths[i] for i in eachindex(env.players)])
	return β
end

export SDGiBS_solve_action
function SDGiBS_solve_action(players::Array, env) 
    error("SDGiBS not implemented yet")
    bₒ = BlockVector([player.history[env.time][2] for player in players],
                    [length(player.history[env.time][2]) for player in players])
    
    ū = BlockVector(vcat([player.predicted_control[env.time:end] for player in players]...),
                    [length(player.predicted_control[env.time:end]) for player in players])

    cₖ = BlockVector([player.cost for player in players],
                    [1 for _ in eachindex(players)])

    cₗ = BlockVector([player.final_cost for player in players],
                    [1 for _ in eachindex(players)])
	
    b̄ = []
    push!(b̄, bₒ)
    normal_distr = MvNormal([0, 0], Matrix(1.0 * I, 2, 2))
    for tt in 1:env.final_time - env.time + 1 # TODO: think abt this one
        # roll out b̄
        push!(b̄, env.state_dynamics(b̄[end], [ū[i][tt] for i in eachindex(players)], rand(2, normal_distr)))
    end

    # Iteration variables
    Q_old = [Inf for _ in eachindex(players)]
    ϵ = 1e-5
    Q_new = cost(players, b̄, ū)

    # Derivative variables
    @variables b̃[1:sum([length(player.history[env.time][2]) for player in players])]
    b = Symbolics.scalarize(b̃)

    # Vaiables to find Wₖ
    @variables m̃[1:env.noise_dim]
    m = Symbolics.scalarize(m̃)
    f = (x, u, m) -> env.state_dynamics(x, u, m)
    Mₖ = Symbolics.gradient(f, m)
    
    while norm(Q_new - Q_old, 2) > ϵ
        # Bakcwards Pass
        # V
        # V_b
        # V_bb    

    end


	return [1, 1]
end

function cost(players, b, u)
    cₖ = BlockVector([player.cost for player in players],
                    [1 for _ in eachindex(players)])

    cₗ = BlockVector([player.final_cost for player in players],
                    [1 for _ in eachindex(players)])

    Q = [0 for _ in eachindex(players)]

    for ii in eachindex(players)
        for tt in eachindex(b[Block(ii)])
            Q[ii] += cₖ[Block(ii)](b[tt][Block(ii)], u[tt][Block(ii)])
        end
    end
    return Q
end


function calculate_belief_variables(env, x̂ₖ, ûₖ)
    #TODO: I'm 100% sure you could make these variables static, no need to re-instantiate...
    @variables x̃[1:env.state_dim], ũ[1:sum([player.action_space for player in players])], m̃[1:env.noise_dim]
    x = Symbolics.scalarize(x̃)
    u = Symbolics.scalarize(ũ)
    m = Symbolics.scalarize(m̃)

    f = (x, u, m) -> env.state_dynamics(x, u, m)
    h = (x, n) -> env.observation_function(;state=x, noise=n)

    x̂ₖ₊₁ = env.state_dynamics(x̂ₖ, uₖ, [0, 0])

    # TODO: noise should be normally distributed

    #TODO: check if this is correct
    Aₖ = substitute.(Symbolics.gradient(f, x), (Dict(x => x̂ₖ, u => uₖ, m => [0, 0]),))
    Mₖ = substitute.(Symbolics.gradient(f, m), (Dict(x => x̂ₖ, u => uₖ, m => [0, 0]),))
    Hₖ = substitute.(Symbolics.gradient(h, x), (Dict(x => x̂ₖ₊₁, n => [0, 0]),))
    Nₖ = substitute.(Symbolics.gradient(h, n), (Dict(x => x̂ₖ₊₁, n => [0, 0]),))

    Γₖ₊₁ = Aₖ * Σₖ * Aₖ' + Mₖ * Mₖ'
    Kₖ = Γₖ₊₁ * Hₖ' * ((Hₖ * Γₖ₊₁ * Hₖ' + Nₖ * Nₖ') \ I)

    x̂ₖ₊₁ = x̂ₖ₊₁ + Kₖ * (observations - h(x̂ₖ₊₁, [0, 0]))
    Σₖ₊₁ = (I - Kₖ * Hₖ) * Γₖ₊₁

    x̂_temp = BlockVector(x̂ₖ₊₁, state_lengths)
    Σ_temp = BlockVector(Σₖ₊₁, cov_lengths)

    β = BlockVector(vcat([vcat(x̂_temp[Block(ii)], Σ_temp[Block(ii)]) for ii in eachindex(env.players)]...), 
                    [state_lengths[i] + cov_lengths[i] for i in eachindex(env.players)])

    return β, x̂ₖ₊₁, Σₖ₊₁, Aₖ, Mₖ, Hₖ, Nₖ, Γₖ₊₁, Kₖ 
end

end
