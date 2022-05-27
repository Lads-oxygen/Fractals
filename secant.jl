using CUDA
# using BenchmarkTools
# using Test
using Adapt
using Serialization
using Plots
# using Polynomials
# using Roots

N = 20
const MAXITER = 20
const TOL = 1e-10
const origin = ComplexF32(-1, 1)
const width = abs2(origin)
const Δ = width / (N - 1)

poly = Vector{ComplexF32}([-1, 0, 0, 1])


function eval(f, x)
    f_x = last(f)
    for i = length(f)-1:-1:1
        @inbounds f_x = f_x * x + f[i]
    end
    return f_x
end


function secant!(f, cfx)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    zm = complex(0)
    zn = origin + complex(x * Δ, -y * Δ)
    f_zm = f[1]
    f_zn = eval(f, zn)

    for i = 1:MAXITER
        # @cuprintln(x, " ", y, " fzn ", f_zn.re, " ", f_zn.im)

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
            cfx[x, y] = zn
            return
        end
    end
    # @cuprintln("did not converge")
    nothing
end

f_x = Matrix{ComplexF32}(undef, N, N)

cpoly = cu(poly)
cfx = cu(f_x)

function run1!(cpoly, cfx)
    kernel = @cuda launch = false secant!(cpoly, cfx)
    config = launch_configuration(kernel.fun)
    threads = min(N^2, config.threads)
    blocks = cld(N^2, threads)
    CUDA.@sync begin
        @cuda threads = threads blocks = blocks secant!(cpoly, cfx)
    end
    # Array(cfx)
    copyto!(f_x, cfx)
    return nothing
end


function main()
    run1!(cpoly, cfx)
    # serialize("data", arr)
    return nothing
end

function make_fig()
    # fractal = deserialize("data")
    fractal = f_x
    gr()
    display(fractal)
    img = heatmap(z=fractal, legend=:none, axis=nothing, size=size(fractal))
    savefig(img, "./fractal.png")
end

main()
make_fig()


# run!(a, f)
# @time run!(a, f)

# print(a, b)






# bench_secant!(b)  # run it once to force compilation
# CUDA.@time bench_secant!(b)
# println(b)



