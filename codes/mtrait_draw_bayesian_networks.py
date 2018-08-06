
#-------------------------------------------Loading libraries------------------------------------------------#

# Loading packages:

from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)
import daft


#---------------------------------------Draw the Bayesian network--------------------------------------------#

# Prefix of the directory of the project is in:
prefix_proj = "/workdir/jp2476/repo/sorghum-multi-trait/"



# Instantiate the PGM.
bn = daft.PGM([2.3, 2.05], origin=[0.3, 0.3])

# Hierarchical parameters.
bn.add_node(daft.Node("alpha", r"$\alpha$", 0.5, 2, fixed=True))
bn.add_node(daft.Node("beta", r"$\beta$", 1.5, 2))

# Latent variable.
bn.add_node(daft.Node("w", r"$w_n$", 1, 1))

# Data.
bn.add_node(daft.Node("x", r"$x_n$", 2, 1, observed=True))

# Add in the edges.
bn.add_edge("alpha", "beta")
bn.add_edge("beta", "w")
bn.add_edge("w", "x")
bn.add_edge("beta", "x")

# And a plate.
bn.add_plate(daft.Plate([0.5, 0.5, 2, 1], label=r"$n = 1, \cdots, N$",
    shift=-0.1))

# Render and save.
bn.render()
bn.figure.savefig("classic.pdf")
bn.figure.savefig("classic.png", dpi=150)