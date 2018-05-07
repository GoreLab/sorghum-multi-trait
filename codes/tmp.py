

import argparse

parser = argparse.ArgumentParser()

#-db DATABASE -u USERNAME -p PASSWORD -size 20000
parser.add_argument("-m", "--model", dest = "model", default = "bn", help="Name of the model")

args = parser.parse_args()

print( "Model: {}".format(
        args.model + ' works!',
        ))