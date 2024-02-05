import MeshCat as mc

abstract type AbstractModel end
abstract type ContinuousDynamicsModel <: AbstractModel end

function RotX(θ)
    [1    0      0
     0  cos(θ) -sin(θ)
     0  sin(θ)  cos(θ)]
end

include("cart_triple_pendulum.jl")


function dynamics_rk4(model::ContinuousDynamicsModel, x::Vector{<:Real}, u::Real, Δt::Real)
    # RK4 with zero-order hold
    k1 = Δt * dynamics(model, x,        u)
    k2 = Δt * dynamics(model, x + k1/2, u)
    k3 = Δt * dynamics(model, x + k2/2, u)
    k4 = Δt * dynamics(model, x + k3,   u)
    return x + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
end

function animate!(vis::mc.Visualizer, model::AbstractModel, X::Vector{<:Vector{<:Real}}, Δt::Real)
    anim = mc.Animation(floor(Int,1/Δt))
    for k = 1:length(X)
        mc.atframe(anim, k) do
            visualize!(vis, model, X[k])
        end
    end
    mc.setanimation!(vis, anim)
end