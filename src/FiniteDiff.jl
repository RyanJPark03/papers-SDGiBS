export finite_diff
function finite_diff(f, x; ϵ = 1e-8)
    num_dimensions = length(x)
    unperturbed_slice = f(x)
    slice_size = size(unperturbed_slice)
    grad::Array{ComplexF64} = Complex.(zeros(tuple(slice_size..., num_dimensions)))

    n = norm(x)

    x_perturbed = copy(x)
    back_perturbed = copy(unperturbed_slice)
    forward_perturbed = copy(unperturbed_slice)
    for i in 1 : num_dimensions
        perturbation = ϵ * x[i] / n
        x_perturbed[i] += perturbation
        back_perturbed = f(x_perturbed)
        x_perturbed[i] -= 2 * perturbation
        forward_perturbed = f(x_perturbed)
        x_perturbed[i] += perturbation

        grad[:, :, i] = (back_perturbed - forward_perturbed) / (2ϵ)
    end

    return grad
end