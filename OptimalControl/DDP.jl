function A_func(state_trans_func, xₖ, uₖ)
    FD.jacobian(_x->state_trans_func(_x, uₖ), xₖ)
end
function B_func(state_trans_func, xₖ, uₖ)
    FD.jacobian(_u->state_trans_func(xₖ, _u), uₖ)
end
function ∂Aᵀ∂x_func(state_trans_func, xₖ, uₖ)
    FD.jacobian(_x->vec(transpose(A_func(state_trans_func, _x, uₖ))), xₖ)
end
function ∂Bᵀ∂u_func(state_trans_func, xₖ, uₖ)
    FD.jacobian(_u->vec(transpose(B_func(state_trans_func, xₖ, _u))), uₖ)
end
function ∂Bᵀ∂x_func(state_trans_func, xₖ, uₖ)
    FD.jacobian(_x->vec(transpose(B_func(state_trans_func, _x, uₖ))), xₖ)
end

function ∇ₓℓ_func(stage_cost, k, xₖ, uₖ)
    FD.gradient(_x->stage_cost(k,_x,uₖ), xₖ)
end
function ∇ᵤℓ_func(stage_cost, k, xₖ, uₖ)
    FD.gradient(_u->stage_cost(k,xₖ,_u), uₖ)
end
function ∇ₓₓℓ_func(stage_cost, k, xₖ, uₖ)
    FD.hessian(_x->stage_cost(k,_x,uₖ), xₖ)
end
function ∇ᵤᵤℓ_func(stage_cost, k, xₖ, uₖ)
    FD.hessian(_u->stage_cost(k,xₖ,_u), uₖ)
end
function ∇ᵤₓℓ_func(stage_cost, k, xₖ, uₖ)
    FD.jacobian(_x->∇ᵤℓ_func(stage_cost,k,_x,uₖ), xₖ)
end

function compute_cost(stage_cost, terminal_cost, x, u)
    @assert length(x) - 1 == length(u)
    cost = terminal_cost(x[end])
    for k = 1:length(u)
        cost += stage_cost(k, x[k], u[k])
    end
    return cost
end

function DDP(state_trans_func, stage_cost, terminal_cost, x₀::SVector{Size_x,<:Real}, u_guess::AbstractVector{<:SVector{Size_u,<:Real}}; tol=1e-10, max_iters=500) where {Size_x,Size_u}

N = length(u_guess) + 1

# forward pass of initial trajectory
x = [zeros(SVector{Size_x}) for k = 1:N]; x[1] = x₀
u = Vector(u_guess)
for k = 1:N-1
    x[k+1] = state_trans_func(x[k], u[k])
end
J = compute_cost(stage_cost, terminal_cost, x, u)

# preallocate for next trajectory
x̆ = [zeros(SVector{Size_x}) for k = 1:N]; x̆[1] = x₀
ŭ = [zeros(SVector{Size_u}) for k = 1:N-1]

# main loop
d = [zeros(SVector{Size_u})        for k = 1:N-1]
K = [zeros(SMatrix{Size_u,Size_x}) for k = 1:N-1]
iter = 0
while iter < max_iters
    iter += 1
    # backward pass
    p = FD.gradient(terminal_cost, x[end])
    P = FD.hessian(terminal_cost, x[end])

    d_norm = 0.0
    ΔJ = 0.0
    for k = N-1:-1:1
        A = A_func(state_trans_func, x[k], u[k])
        B = B_func(state_trans_func, x[k], u[k])

        gₓ = ∇ₓℓ_func(stage_cost, k, x[k], u[k]) + A' * p
        gᵤ = ∇ᵤℓ_func(stage_cost, k, x[k], u[k]) + B' * p

        Gₓₓ = ∇ₓₓℓ_func(stage_cost, k, x[k], u[k]) + A' * P * A
        Gᵤᵤ = ∇ᵤᵤℓ_func(stage_cost, k, x[k], u[k]) + B' * P * B
        Gᵤₓ = ∇ᵤₓℓ_func(stage_cost, k, x[k], u[k]) + B' * P * A

        β = 0.1
        while !isposdef(Gᵤᵤ)
            Gᵤᵤ += β * I
            β *= 2
        end

        dₖ = Gᵤᵤ \ gᵤ
        Kₖ = Gᵤᵤ \ Gᵤₓ

        p = gₓ - Kₖ'gᵤ + Kₖ'Gᵤᵤ*dₖ - Gᵤₓ'dₖ
        P = Gₓₓ + Kₖ'Gᵤᵤ*Kₖ - 2Kₖ'Gᵤₓ

        d[k] = dₖ
        K[k] = Kₖ
        d_norm = max(d_norm, maximum(abs.(dₖ)))
        ΔJ += gᵤ'dₖ
    end

    # Armijo line search
    b = 0.001 # Armijo parameter
    α = 1
    J̆ = Inf
    while J̆ > J - b * α * ΔJ
        for k = 1:N-1
            ŭ[k] = u[k] - α * d[k] - K[k] * (x̆[k] - x[k])
            x̆[k+1] = state_trans_func(x̆[k], ŭ[k])
        end
        J̆ = compute_cost(stage_cost, terminal_cost, x̆, ŭ)
        α = α * 0.5
    end
    x, x̆ = x̆, x
    u, ŭ = ŭ, u
    J = J̆

    # finish if gradient less than tolerance
    if d_norm < tol
        break
    end
end

if iter < max_iters
    @info "DDP completed in $iter iterations because gradient is less than tolerance"
else
    @info "DDP completed because max_iters was reached"
end

return (x=x, u=u, K=K)

end