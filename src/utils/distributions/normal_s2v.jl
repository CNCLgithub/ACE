export normal_s2v, Normal_S2V

struct Normal_S2V <: Gen.Distribution{S2V} end

const normal_s2v = Normal_S2V()

function Gen.random(::Normal_S2V, mus::S2V, sigma::Float64)
    S2V(
        normal(mus[1], sigma),
        normal(mus[2], sigma)
    )
end

function Gen.logpdf(::Normal_S2V, x::S2V, mu::S2V, sigma::Float64)
    logpdf(normal, x[1], mu[1], sigma) + logpdf(normal, x[2], mu[2], sigma)
end

(::Normal_S2V)(mu, sigma) = Gen.random(normal_s2v, mu, sigma)

Gen.has_output_grad(::Normal_S2V) = true
function Gen.logpdf_grad(::Normal_S2V, x::S2V, mu::S2V, sigma::Float64)
    (deriv_x_1, deriv_mu_1, deriv_std_1) = logpdf_grad(normal, x[1], mu[1], sigma)
    (deriv_x_2, deriv_mu_2, deriv_std_2) = logpdf_grad(normal, x[1], mu[1], sigma)
    S2V(deriv_x_1, deriv_x_2), S2V(deriv_mu_1, deriv_mu_2), deriv_std_1 + deriv_std_2
end
