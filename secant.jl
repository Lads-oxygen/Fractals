using CUDA
# using BenchmarkTools
# using Test
# using Adapt
# using Plots
using Polynomials
# using Roots
using GLMakie

N = 512
const MAXITER = 50
const TOL = 1e-20

poly = Vector{ComplexF32}([-1, 0, 0, 1])
p = Polynomial(poly)
roots = Polynomials.roots(p)
f_x = Matrix{ComplexF32}(undef, N, N)


function eval(f, x)
    f_x = last(f)
    for i = length(f)-1:-1:1
        @inbounds f_x = f_x * x + f[i]
    end
    return f_x
end


function secant!(f, croots, origin, Δ, cfractal)

    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    zm = complex(0)
    zn = origin + complex(x * Δ, -y * Δ)
    f_zm = f[1]
    f_zn = eval(f, zn)
    # @cuprintln(x, " ", y, " fzn ", zn.re, " ", zn.im)

    for i = 1:MAXITER
    
        if abs(f_zn - f_zm) < TOL
            # @cuprintln("division by 0")
            return
        end
    
        temp = zn - f_zn * (zn - zm) / (f_zn - f_zm)
        zm = zn
        zn = temp
        f_zm = f_zn
        f_zn = eval(f, zn)
    
        if abs(zn - zm) < TOL && abs(f_zn) < TOL
            # @cuprintln("root is", zn.re, " ", zn.im)
            break
        end

        if i == MAXITER
            return
        end
    end


    for i = 1:length(croots)
        if abs(zn - croots[i]) < 1e-5
            cfractal[x, y] = i
            break
        end
    end        
    nothing
end


cpoly = cu(poly)
croots = cu(roots)
cfx = cu(f_x)
fractal = zeros(Int8, N, N)
cfractal = cu(fractal)


function run!(cpoly, croots, origin, cfractal)
    width = abs2(origin)
    Δ = width / (N - 1)
    display(origin)
    threads = min(N, 8)
    blocks = cld(N, threads)
    CUDA.@sync begin
        @cuda threads = (threads, threads) blocks = (blocks, blocks) secant!(cpoly, croots, origin, Δ, cfractal)
    end
    # println(threads, " ", blocks)
    return nothing
end

function update(origin, cfractal)
    run!(cpoly, croots, origin, cfractal)
    Array(cfractal)
end

function main()
    origin = ComplexF32(-1, 1)
    fractal = update(origin, cfractal)
    canvas(fractal)
end



function canvas(fractal)
    origin = complex(1,1)
    # f = Observable(fractal)
    GLMakie.activate!()
    set_window_config!(;
        vsync=false,
        framerate=30.0,
        float=false,
        pause_rendering=false,
        focus_on_show=false,
        decorated=true,
        title="Fractal"
    )
    scene = Scene()
    subwindow = Scene(scene, px_area=Rect(0, 0, 300, 300), clear=true, backgroundcolor=:black)
    fig, axis, plot = heatmap(fractal)
    display(fig)
    for i = 1:1000
        origin *= 0.99
        plot[1] = update!(origin, cfractal)
        yield()
        sleep(3)
    end
        

end



main()



# @time
# @btime
# CUDA.@time


