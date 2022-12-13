
weight_file=models/weights
mkdir $weight_file

wget https://github.com/vbalnt/tfeat/blob/master/pretrained-models/tfeat-liberty.params -O weight_file/tfeat-liberty.params
wget https://github.com/scape-research/SOSNet/blob/master/sosnet-weights/sosnet-32x32-liberty.pth -O $weight_file/sosnet-32x32-liberty.pth

# From Google Drive
curl https://drive.google.com/u/0/uc?id=1NDJuzo6SpIIYCfdMlWosJSvaCsPZRicW&export=download >$weight_file/unet-thu_seg.pth

# The weight files are provided in Baidu Cloud as well.
# See https://pan.baidu.com/s/1DsOx1rHZuUROQLm-EleFhQ?pwd=srqu