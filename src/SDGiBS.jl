module SDGiBS

using BlockArrays

export belief_update
function belief_update(env::base_environment, players::Array{players}, observations::BlockVector{Float64})
    β = BlockVector([player.belief for player in players], [length(player.belief) for player in players])

    x̂ₖ = BlockVector(vcat([β[Block(ii)][1:env.state_dim] for ii in eachindex(env.players)]),
            [env.state_dim for _ in eachindex(env.players)])
    Σₖ = BlockVector(vcat([β[Block(ii)][env.state_dim+1:end] for ii in eachindex(env.players)]),
            [length(β[Block(1)]) - env.state_dim for _ in eachindex(env.players)])

    @variables x̃[1:(parameter_dimension)], ũ[1:]
    x = Symbolics.scalarize(x̃)

    Aₖ = Symbolics.gradient(env.state_dynamics, x, u)

    Γₖ₊₁ = 

	return belief
end

export SDGiBS_solve_action
function SDGiBS_solve_action(players::Array{T1}, env::T2)
	# TODO: datatype T must have a "belief," "cost" 
	return [1, 1]
end

end
