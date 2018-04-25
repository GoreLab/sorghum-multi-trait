

########################################Spalla###########################################

## Uploading the folder content [Whole folder]:
gdrive sync upload --delete-extraneous --timeout 10000000 /home/jhonathan/Documentos/resul_mtrait-proj \
 1N45DjnYKcAmQiOpX_4Xg9gESyfihXikN

## Downloading the folder content [Kaggle submission]:
gdrive sync download --delete-extraneous  --timeout 10000000 1N45DjnYKcAmQiOpX_4Xg9gESyfihXikN \
/home/jhonathan/Documentos/resul_mtrait-proj



########################################falc0n###########################################

## Creating a directory at google drive (only for the first time in the first pc):
gdrive mkdir resul_mtrait-proj

# Output --> Directory 1N45DjnYKcAmQiOpX_4Xg9gESyfihXikN created

## Uploading the folder content:
gdrive sync upload --delete-extraneous --timeout 10000000 /home/jhonathan/Documents/resul_mtrait-proj \
1N45DjnYKcAmQiOpX_4Xg9gESyfihXikN

## Downloading the folder content:
gdrive sync download --delete-extraneous --timeout 10000000 1N45DjnYKcAmQiOpX_4Xg9gESyfihXikN \
/home/jhonathan/Documents/resul_mtrait-proj


####################################### Dayhoff1 #########################################

## Uploading the folder content [Whole folder]:
/data1/aafgarci/jhonathan/sorghum-multi-trait/codes/gdrive sync upload --delete-extraneous --timeout 10000000 /data1/aafgarci/jhonathan/resul_mtrait-proj \
 1N45DjnYKcAmQiOpX_4Xg9gESyfihXikN

## Downloading the folder content [Kaggle submission]:
/data1/aafgarci/jhonathan/sorghum-multi-trait/codes/gdrive sync download --delete-extraneous  --timeout 10000000 1N45DjnYKcAmQiOpX_4Xg9gESyfihXikN \
/data1/aafgarci/jhonathan/resul_mtrait-proj

####################################### cbsugore02 #########################################

## Uploading the folder content [Whole folder]:
/workdir/jp2476/repo/sorghum-multi-trait/codes/gdrive sync upload --delete-extraneous --timeout 10000000 /workdir/jp2476/repo/resul_mtrait-proj \
 1N45DjnYKcAmQiOpX_4Xg9gESyfihXikN

## Downloading the folder content [Kaggle submission]:
/workdir/jp2476/repo/sorghum-multi-trait/codes/gdrive sync download --delete-extraneous  --timeout 10000000 1N45DjnYKcAmQiOpX_4Xg9gESyfihXikN \
/workdir/jp2476/repo/resul_mtrait-proj

############################ Install python locally ####################################


# For install python local:
bash ./Anaconda3-5.0.1-Linux-x86_64.sh

# Folder where the python is installed:
/workdir/jp2476/software/python

# To install modules:
/workdir/jp2476/software/python/bin/pip install SOFTWARE_NAME --upgrade

# To fire bash:
/workdir/jp2476/software/python/bin/python/bash_2layers.py > out.txt

# Bug instruction:
# when saving the output from the tensorflow, define the dir first, then do ./ on the dir arg of tf_saver

#########################################################################################




## On going analysi