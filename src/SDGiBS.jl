module SDGiBS

using BlockArrays

export belief_update
function belief_update(β :: BlockVector{Float64}, observation :: BlockVector{Float64}) 
    
    # all_beliefs = BlockVector{Float64}(vcat([player.belief for player in players]...), [length(player.belief) for player in players])
    # TODO: Implement belief update
    return belief
end

export SDGiBS_solve_action
function SDGiBS_solve_action(players :: Array{T1}, env :: T2, current_player_index :: Int)
    # TODO: datatype T must have a "belief," "cost" 
    return [1, 1]
end

end 
