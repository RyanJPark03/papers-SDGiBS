module SDGiBS

using BlockArrays
using LinearAlgebra
using Enzyme

Enzyme.API.printunnecessary!(false)
Enzyme.API.runtimeActivity!(true) 
Enzyme.API.strictAliasing!(false)

export belief_update
function belief_update(env, players::Array, observations)
    num_players = length(players)
    β = BlockVector(vcat([player.belief for player in players]...), [length(player.belief) for player in players])
    # β = vcat([player.belief for player in players]...)

    mean_lengths = [env.state_dim for _ in 1:num_players]
    cov_length = Int(sqrt(Int(length(β) // num_players - env.state_dim)))
    cov_lengths = [cov_length for _ in 1:num_players]

    x̂ₖ = BlockVector(vcat([β[Block(ii)][1:env.state_dim] for ii in 1:num_players]...), mean_lengths)
    # x̂ₖ = vcat([player.belief[1:env.state_dim] for player in players]...)
    # x̂ₖ_vec = Vector(x̂ₖ)
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
    uₖ = BlockVector(vcat([player.predicted_control[env.time] for player in players]...), 
                    [player.action_space for player in players])
    m = BlockVector([0.0 for _ in 1:env.dynamics_noise_dim * num_players],
                    [env.dynamics_noise_dim for _ in 1:num_players])

    f = (x) -> env.state_dynamics(
        BlockVector(x[Block(1)], mean_lengths),
        BlockVector(x[Block(2)], [player.action_space for player in players]),
        BlockArray(x[Block(3)], [env.dynamics_noise_dim for _ in 1:num_players]))
        
    h = (x) -> env.observation_function(
        states = BlockVector(x[Block(1)], mean_lengths),
        m = BlockVector(x[Block(2)], [env.observation_noise_dim for _ in 1:num_players]))


    # f_jacobian = Enzyme.jacobian(Forward, f, [x̂ₖ, uₖ, m])
    # TODO: there are NaNs and Infs in the jacoban
    f_jacobian = Enzyme.jacobian(Forward, f, BlockVector(vcat([x̂ₖ, uₖ, m]...), [length(x̂ₖ), length(uₖ), length(m)]))
    Aₖ = f_jacobian[:, 1:length(x̂ₖ)]
    Mₖ = f_jacobian[:, length(x̂ₖ) + length(uₖ)+ 1:end]

    n = BlockVector([0.0 for _ in 1:env.observation_noise_dim * num_players],
                    [env.observation_noise_dim for _ in 1:num_players])

    # asdf = (x) -> x

    temp = BlockVector(vcat([x̂ₖ, n]...), [length(x̂ₖ), length(n)])
    # h_jacobian = Enzyme.jacobian(Forward, h, [1 for _ in 1 : length(x̂ₖ) + length(n)])
    h_jacobian = Enzyme.jacobian(Forward, h, temp)
    # h_jacobian = Enzyme.jacobian(Forward, env.observation_function, x̂ₖ, n)
    # h_jacobian = Enzyme.autodiff(Forward, env.observation_function, x̂ₖ, n)



    Hₖ = h_jacobian[:, 1:length(x̂ₖ)]
    Nₖ = h_jacobian[:, length(x̂ₖ) + 1:end]

    println("working 1")
    display(h_jacobian)
    println("now f ")
    # println(f_jacobian)
    display(Aₖ)
    display(Mₖ)

    Γₖ₊₁ = Aₖ * Σₖ * Aₖ' + Mₖ * Mₖ'
    Kₖ = Γₖ₊₁ * Hₖ' * ((Hₖ * Γₖ₊₁ * Hₖ' + Nₖ * Nₖ') \ I)

    x̂ₖ₊₁ = env.state_dynamics(x̂ₖ, u, m) + Kₖ * (observations - h(x̂ₖ₊₁, [0.0 for _ in 1:env.observation_noise_dim * num_players]))
    Σₖ₊₁ = (I - Kₖ * Hₖ) * Γₖ₊₁

    println("working 2")
    @assert typeof(x̂ₖ₊₁) == Vector{Float64}

    x̂_temp = BlockVector(x̂ₖ₊₁, state_lengths)
    Σ_temp = BlockVector(Σₖ₊₁, cov_lengths)

    β = BlockVector(vcat([vcat(x̂_temp[Block(ii)], Σ_temp[Block(ii)]) for ii in eachindex(env.players)]...), 
                    [state_lengths[i] + cov_lengths[i] for i in eachindex(env.players)])

    println("working 3")
    # β = vcat([vcat(x̂ₖ₊₁, vec(Σₖ₊₁)) for ii in eachindex(players)]...)
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

    # # Derivative variables
    # @variables b̃[1:sum([length(player.history[env.time][2]) for player in players])]
    # b = Symbolics.scalarize(b̃)

    # # Vaiables to find Wₖ
    # @variables m̃[1:env.noise_dim]
    # m = Symbolics.scalarize(m̃)
    # f = (x, u, m) -> env.state_dynamics(x, u, m)
    # Mₖ = Symbolics.gradient(f, m)
    
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
end

end
