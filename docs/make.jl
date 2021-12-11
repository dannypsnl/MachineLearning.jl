using Documenter
using MachineLearning

makedocs(
    sitename = "MachineLearning",
    format = Documenter.HTML(),
    modules = [MachineLearning]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
