export ReceptiveFields

# HINT: see https://www.gen.dev/docs/stable/how_to/custom_distributions/#Defining-New-Distributions-From-Scratch

# `Ref{Py}` is a pointer to some python object
# In this case (hopefully) a jax vector of float
struct ReceptiveFields <: Distribution{Ref{Py}} end

const receptive_fields = ReceptiveFields()

# TODO: Implement the following methods using the interface to python

function random(::ReceptiveFields, rfs::Ref{Py}, scene::Ref{Py})
end

(::ReceptiveFields, rfs::Ref{Py}, scene::Ref{Py}) =
    random(ReceptiveFields(), rfs, scene)

function logpdf(::ReceptiveFields, x::Ref{Py}, rfs::Ref{Py}, scene::Ref{Py})
end

# TODO: we could implement for gradient methods but maybe later
function logpdf_grad(::ReceptiveFields, x::Ref{Py}, rfs::Ref{Py}, scene::Ref{Py})
end

has_argument_grads(::ReceptiveFields) = (false,)
has_output_grad(::ReceptiveFields) = false
is_discrete(::ReceptiveFields) = false
    


