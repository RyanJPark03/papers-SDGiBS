module SDGiBS

using BlockArrays

export belief_update
function belief_update(env::base_environment, players::Array{players}, observations::BlockVector{Float64})
    β = BlockVector([player.belief for player in players], [length(player.belief) for player in players])

    mean_lengths = [env.state_dim for _ in eachindex(env.players)]
    cov_lengths = [length(β[Block(1)]) - env.state_dim for _ in eachindex(env.players)]

    x̂ₖ = BlockVector(vcat([β[Block(ii)][1:env.state_dim] for ii in eachindex(env.players)]), mean_lengths)
    Σₖ = BlockVector(vcat([β[Block(ii)][env.state_dim+1:end] for ii in eachindex(env.players)]), cov_lengths)
    uₖ = BlockVector([player.predicted_control[end][end] for player in players], 
                    [player.action_space for player in players])

    @variables x̃[1:env.state_dim], ũ[1:sum([player.action_space for player in players])], m̃[1:env.noise_dim]
    x = Symbolics.scalarize(x̃)
    u = Symbolics.scalarize(ũ)
    m = Symbolics.scalarize(m̃)

    f = (x, u, m) -> env.state_dynamics(x, u, m)
    h = (x, n) -> env.observation_function(;state=x, noise=n)

    x̂ₖ₊₁ = env.state_dynamics(x̂ₖ, uₖ, [0, 0])

    # TODO: noise should be normally distributed

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

    β = BlockVector([vcat(x̂_temp[Block(ii)], Σ_temp[Block(ii)]) for ii in eachindex(env.players)]..., 
                    [state_lengths[i] + cov_lengths[i] for i in eachindex(env.players)])
	return β
end

export SDGiBS_solve_action
function SDGiBS_solve_action(players::Array{T1}, env::T2)
	# TODO: datatype T must have a "belief," "cost" 
	return [1, 1]
end

end
