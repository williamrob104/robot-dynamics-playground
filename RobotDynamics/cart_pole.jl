Base.@kwdef struct CartPole <: ContinuousDynamicsModel
    m0::Float64 = 1.0
    m::Float64 = 0.2
    ℓ::Float64 = 0.5
    input::Symbol = :force
end

function dynamics(model::CartPole, x::Vector{<:Real}, u::Real)
    s, θ, ṡ, θ̇ = x

    m0, m, ℓ, g = model.m0, model.m, model.ℓ, 9.81

    M = [m+m0  m*ℓ*cos(θ); m*ℓ*cos(θ) m*ℓ^2]
    f = [m*ℓ*θ̇^2*sin(θ) + u; -m*g*ℓ*sin(θ)]
    if model.input == :acceleration
        M[1,:] = [1,0]
        f[1] = u
    end
    s̈, θ̈ = M \ f

    return [ṡ; θ̇; s̈; θ̈]
end

function set_mesh!(vis::mc.Visualizer, model::CartPole)
    cart_w, cart_h = 0.2, 0.15
    cart = mc.HyperRectangle(mc.Vec(-cart_h,-cart_w/2,-cart_h/2), mc.Vec(cart_h,cart_w,cart_h))
    mc.setobject!(vis[:cart_pendulum][:cart], cart, mc.MeshPhongMaterial(color=mc.RGB(0,0,0)))

    rail_l, rail_h = 50, 0.02
    rail = mc.HyperRectangle(mc.Vec(-cart_h,-rail_l/2,-cart_h/2-rail_h), mc.Vec(cart_h,rail_l,rail_h))
    mc.setobject!(vis[:cart_pendulum][:rail], rail, mc.MeshPhongMaterial(color=mc.RGB(0.5,0.5,0.5)))

    w = 0.08
    hinge = mc.Cylinder(mc.Point(0.,0,0), mc.Point(w,0,0), w/2)
    pole  = mc.Cylinder(mc.Point(w/2,0,model.ℓ/2), mc.Point(w/2,0,-model.ℓ/2), w/4)
    mass  = mc.HyperSphere(mc.Point(w/2,0,0), 0.1)
    mc.setobject!(vis[:cart_pendulum][:pendulum][:hinge], hinge, mc.MeshPhongMaterial(color=mc.RGB(0,1,0)))
    mc.setobject!(vis[:cart_pendulum][:pendulum][:pole],  pole,  mc.MeshPhongMaterial(color=mc.RGB(0,1,0)))
    mc.setobject!(vis[:cart_pendulum][:pendulum][:mass],  mass,  mc.MeshPhongMaterial(color=mc.RGB(1,0,0)))

    mc.setprop!(vis["/Grid"], "visible", false)
    mc.setprop!(vis["/Axes"], "visible", false)

    visualize!(vis, model, [0,0])
end

function visualize!(vis::mc.Visualizer, model::CartPole, x::Vector)
    s, θ = x

    y = s
    z = 0.
    mc.settransform!(vis[:cart_pendulum][:cart], mc.Translation(0,y,z))
    mc.settransform!(vis[:cart_pendulum][:pendulum][:hinge], mc.Translation(0,y,z))
    y += model.ℓ * sin(θ) / 2
    z -= model.ℓ * cos(θ) / 2
    mc.settransform!(vis[:cart_pendulum][:pendulum][:pole], mc.Translation(0,y,z) ∘ mc.LinearMap(RotX(θ)))
    y += model.ℓ * sin(θ) / 2
    z -= model.ℓ * cos(θ) / 2
    mc.settransform!(vis[:cart_pendulum][:pendulum][:mass], mc.Translation(0,y,z))

end