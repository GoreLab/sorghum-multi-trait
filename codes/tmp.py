

import argparse

parser = argparse.ArgumentParser()

#-db DATABASE -u USERNAME -p PASSWORD -size 20000
parser.add_argument("-m", "--model", dest = "model", default = "bn", help="Name of the model")
parser.add_argument("-cv", "--cv", dest = "cv", default = "cv1", help="Cross-validation type")

args = parser.parse_args()

print( "Model: {} Type of cv: {}".format(
        args.model + ' works!',
        args.cv + ' works!'       
        ))