struct CartTriplePendulum <: ContinuousDynamicsModel
    m0::Float64
    m1::Float64
    m2::Float64
    m3::Float64
    ℓ1::Float64
    ℓ2::Float64
    ℓ3::Float64
    a1::Float64
    a2::Float64
    a3::Float64
    J1::Float64
    J2::Float64
    J3::Float64
    d1::Float64
    d2::Float64
    d3::Float64
    input::Symbol
    C1::Matrix{Float64}
    C2::Vector{Float64}
    D::Matrix{Float64}

    CartTriplePendulum(m0, m1,m2,m3, ℓ1,ℓ2,ℓ3, a1,a2,a3, J1,J2,J3, d1,d2,d3, input) = ~(input == :force || input == :acceleration) ?
    error("input must be :force or :acceleration") :
    new(m0, m1,m2,m3, ℓ1,ℓ2,ℓ3, a1,a2,a3, J1,J2,J3, d1,d2,d3, input,
        [J1 + m1*a1^2 + m2*ℓ1^2 + m3*ℓ1^2    m2*ℓ1*a2 + m3*ℓ1*ℓ2       m3*ℓ1*a3
               m2*ℓ1*a2 + m3*ℓ1*ℓ2          J2 + m2*a2^2 + m3*ℓ2^2     m3*ℓ2*a3
                     m3*ℓ1*a3                      m3*ℓ2*a3          J3 + m3*a3^2],
        [m1*a1+m2*ℓ1+m3*ℓ1;   m2*a2+m3*ℓ2;   m3*a3],
        [d1+d2  -d2   0
          -d2  d2+d3 -d3
           0    -d3   d3])
end


CartTriplePendulum(;m0=0, m1=1,m2=1,m3=1, ℓ1=.5,ℓ2=.5,ℓ3=.6, a1=.25,a2=.25,a3=.25, J1=0.5^2/12,J2=0.5^2/12,J3=0.5^2/12, d1=0,d2=0,d3=0, input=:force) = CartTriplePendulum(m0, m1,m2,m3, ℓ1,ℓ2,ℓ3, a1,a2,a3, J1,J2,J3, d1,d2,d3, input)

CartTriplePendulum(ℓ1, ℓ2, ℓ3; m0=0, d1=0, d2=0, d3=0, input=:force) = CartTriplePendulum(m0, ℓ1*12,ℓ2*12,ℓ3*12, ℓ1,ℓ2,ℓ3, ℓ1/2,ℓ2/2,ℓ3/2, ℓ1^3,ℓ2^3,ℓ3^3, d1,d2,d3, input)

function dynamics(model::CartTriplePendulum, x::Vector{<:Real}, u::Real)
    s = x[1]
    θ = x[2:4]
    ṡ = x[5]
    θ̇ = x[6:8]
    g = 9.81

    C1, C2, D = model.C1, model.C2, model.D

    Δθ = [θ[i]-θ[j] for i=1:3, j=1:3]

    if model.input == :acceleration
        s̈ = u
    else
        s̈ = (u + (C2 .* sin.(θ))' * (θ̇.^2)) / (model.m0+model.m1+model.m2+model.m3)
    end

    θ̈ = -(C1 .* cos.(Δθ)) \ (C2 .* cos.(θ) * s̈ + (C1 .* sin.(Δθ)) * θ̇.^2 + D * θ̇ + C2 .* sin.(θ) * g)
    return [ṡ; θ̇; s̈; θ̈]
end

function set_mesh!(vis::mc.Visualizer, model::CartTriplePendulum)
    cart_w, cart_h = 0.2, 0.15
    cart = mc.HyperRectangle(mc.Vec(-cart_h,-cart_w/2,-cart_h/2), mc.Vec(cart_h,cart_w,cart_h))
    mc.setobject!(vis[:cart_triple_pendulum][:cart], cart, mc.MeshPhongMaterial(color=mc.RGB(0,0,0)))

    rail_l, rail_h = 50, 0.02
    rail = mc.HyperRectangle(mc.Vec(-cart_h,-rail_l/2,-cart_h/2-rail_h), mc.Vec(cart_h,rail_l,rail_h))
    mc.setobject!(vis[:cart_triple_pendulum][:rail], rail, mc.MeshPhongMaterial(color=mc.RGB(0.5,0.5,0.5)))

    t, w = 0.04, 0.08
    mat = [mc.MeshPhongMaterial(color=mc.RGB(1,0,0)),
           mc.MeshPhongMaterial(color=mc.RGB(0,1,0)),
           mc.MeshPhongMaterial(color=mc.RGB(0,0,1))]
    ℓ = [model.ℓ1, model.ℓ2, model.ℓ3]
    for i = 1:3
        hinge = mc.Cylinder(mc.Point(t*(i-1),0,0), mc.Point(t*i,0,0), w/2)
        body  = mc.HyperRectangle(mc.Vec(t*(i-1),-w/2,-ℓ[i]/2), mc.Vec(t,w,ℓ[i]))
        mc.setobject!(vis[:cart_triple_pendulum]["link$i"][:head], hinge, mat[i])
        mc.setobject!(vis[:cart_triple_pendulum]["link$i"][:body], body,  mat[i])
        mc.setobject!(vis[:cart_triple_pendulum]["link$i"][:tail], hinge, mat[i])
    end
    visualize!(vis, model, [0,0,0,0])

    mc.setprop!(vis["/Grid"], "visible", false)
    mc.setprop!(vis["/Axes"], "visible", false)
end

function visualize!(vis::mc.Visualizer, model::CartTriplePendulum, x::Vector)
    s = x[1]
    θ = x[2:4]

    y = s
    z = 0.
    mc.settransform!(vis[:cart_triple_pendulum][:cart], mc.Translation(0,y,z))

    ℓ = [model.ℓ1, model.ℓ2, model.ℓ3]
    for i = 1:3
        mc.settransform!(vis[:cart_triple_pendulum]["link$i"][:head], mc.Translation(0,y,z))
        y += ℓ[i] * sin(θ[i]) / 2
        z -= ℓ[i] * cos(θ[i]) / 2
        mc.settransform!(vis[:cart_triple_pendulum]["link$i"][:body], mc.Translation(0,y,z) ∘ mc.LinearMap(RotX(θ[i])))
        y += ℓ[i] * sin(θ[i]) / 2
        z -= ℓ[i] * cos(θ[i]) / 2
        mc.settransform!(vis[:cart_triple_pendulum]["link$i"][:tail], mc.Translation(0,y,z))
    end
end