device=0
# rm spmm_test_out.out grb_test_out.out
# echo "data,K=128-cusparse-gflops,K=128-gespmm-gflops,K=256-cusparse-gflops,K=256-gespmm-gflops,K=512-cusparse-gflops,K=512-gespmm-gflops," >> spmm_test_out.out

cd ./cs-spmm

# rm -f csspmm_test.log
# make 

run_csspmm(){
    data=$1
    N=$2
    S=$3
    ./csspmm_test $data $device $N $S
}

DIR="/home/ubuntu/gnn_benchmark/spmm_test/datasets"

# echo "Testing for GCN Settings"
# graph="/home/ubuntu/gnn_benchmark/spmm_test/datasets/pubmed.mtx"
# echo $graph
# echo "N = 32"
# run_csspmm $graph 32 16
# echo "N = 3"
# run_csspmm $graph 3 16
# 
# graph="/home/ubuntu/gnn_benchmark/spmm_test/datasets/ogbn-arxiv.mtx"
# echo $graph
# echo "N = 128"
# run_csspmm $graph 128 16
# echo "N = 256"
# run_csspmm $graph 256 16
# echo "N = 40"
# run_csspmm $graph 40 16
# 
# graph="/home/ubuntu/gnn_benchmark/spmm_test/datasets/ogbn-proteins.mtx"
# echo $graph
# echo "N = 8"
# run_csspmm $graph 8 256
# echo "N = 256"
# run_csspmm $graph 256 256
# echo "N = 112"
# run_csspmm $graph 112 256
# 
graph="/home/ubuntu/gnn_benchmark/spmm_test/datasets/reddit.mtx"
echo $graph
echo "N = 128"
run_csspmm $graph 128 64
echo "N = 41"
run_csspmm $graph 41 64

# graph="$DIR/reddit_s64.mtx"
# echo $graph
# echo "N = 128"
# run_csspmm $graph 128 256
# echo "N = 41"
# run_csspmm $graph 41 256

# graph="$DIR/reddit_s128.mtx"
# echo $graph
# echo "N = 128"
# run_csspmm $graph 128 256
# echo "N = 41"
# run_csspmm $graph 41 256
# 
# graph="$DIR/reddit_s256.mtx"
# echo $graph
# echo "N = 128"
# run_csspmm $graph 128 256
# echo "N = 41"
# run_csspmm $graph 41 256

cat csspmm_test.log

cd ..
