

## For firing python scripts into bash code:

# Prefix python:
PREFIX_python = /workdir/jp2476/software/python/bin

# Prefix code:
PREFIX_code = /workdir/jp2476/repo/sorghum-multi-trait/codes

# Firing the process:
${PREFIX_python}/python ${PREFIX_code}/mtrait_data_processing.py & 
