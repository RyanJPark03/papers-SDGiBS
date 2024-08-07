export finite_diff
function finite_diff(f, x; ϵ = 1e-9)
    num_dimensions = length(x)
    unperturbed_slice = f(x)
    slice_size = size(unperturbed_slice)
    grad = zeros(tuple(slice_size..., num_dimensions))

    n = norm(x)

    # Active surveillance specific
    state_dim = 4
    x_perturbed = copy(x)
    back_perturbed = copy(unperturbed_slice)
    forward_perturbed = copy(unperturbed_slice)
    for player in 1:2
        # Do the belief mean perturbations
        for i in 1:state_dim
            idx = (20 * (player - 1)) + i
            perturbation = ϵ * x[idx] / n
            x_perturbed[idx] += perturbation
            forward_perturbed = f(x_perturbed)
            if any((x) -> imag(x) != 0.0, forward_perturbed)
                @warn "Imaginary values found in Wₖ forward_perturb"
                println("Wₖ forward_perturb")
                show(stdio, "text/plain", forward_perturbed)
                forward_perturbed = abs.(forward_perturbed)
            end

            x_perturbed[idx] -= 2 * perturbation
            back_perturbed = f(x_perturbed)
            if any((x) -> imag(x) != 0.0, back_perturbed)
                @warn "Imaginary values found in Wₖ back_perturb"
                println("Wₖ back_perturb")
                show(stdio, "text/plain", back_perturbed)
                back_perturbed = abs.(back_perturbed)
            end


            grad[:, :, idx] = (back_perturbed - forward_perturbed) / (2ϵ)
        end

        # Do the symmetric perturbations
        for i in 1 : state_dim
            for j in i : state_dim
                idx = (20 * (player - 1)) + 4 + 4 * (i - 1) + j
                mirror_idx = (20 * (player - 1)) + 4 +  4 * (j - 1) + i
                
                perturbation = ϵ * x[idx] / n
                x_perturbed[idx] += perturbation
                x_perturbed[mirror_idx] += perturbation
                forward_perturbed = abs.(f(x_perturbed))
                
                x_perturbed[idx] -= 2 * perturbation
                x_perturbed[mirror_idx] -= 2 * perturbation
                back_perturbed = abs.(f(x_perturbed))

                grad[:, :, idx] = (back_perturbed - forward_perturbed) / (4ϵ)
                grad[:, :, mirror_idx] = (back_perturbed - forward_perturbed) / (4ϵ)
            end
        end

        for i in 1 : 2 # 2 = action dimension
            idx = 40 + (2 * (player - 1)) + i
            perturbation = ϵ * x[idx] / n
            x_perturbed[idx] += perturbation
            forward_perturbed = abs.(f(x_perturbed))

            x_perturbed[idx] -= 2 * perturbation
            back_perturbed = abs.(f(x_perturbed))

            grad[:, :, idx] = (back_perturbed - forward_perturbed) / (2ϵ)
        end
    end

    # x_perturbed = copy(x)
    # back_perturbed = copy(unperturbed_slice)
    # forward_perturbed = copy(unperturbed_slice)
    # for i in 1 : num_dimensions
    #     perturbation = ϵ * x[i] / n
    #     x_perturbed[i] += perturbation
    #     back_perturbed = abs.(f(x_perturbed))
    #     # Main.@infiltrate (any((x) -> imag(x) != 0.0, back_perturbed))

    #     # @assert !(any((x) -> imag(x) != 0.0, back_perturbed))

    #     x_perturbed[i] -= 2 * perturbation
    #     forward_perturbed = abs.(f(x_perturbed))

    #     # @assert !(any((x) -> imag(x) != 0.0, forward_perturbed))

    #     x_perturbed[i] += perturbation

    #     grad[:, :, i] = (back_perturbed - forward_perturbed) / (2ϵ)
    # end

    return grad
end