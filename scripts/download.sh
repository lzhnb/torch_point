SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

cd $SCRIPTPATH/..
wget https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip --no-check-certificate
wget https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip --no-check-certificate
unzip shapenetcore_partanno_segmentation_benchmark_v0.zip
unzip modelnet40_normal_resampled.zip
rm shapenetcore_partanno_segmentation_benchmark_v0.zip
rm modelnet40_normal_resampled.zip
mv shapenetcore_partanno_segmentation_benchmark_v0 ../data
mv modelnet40_normal_resampled ../data
cd -