device=0
# rm spmm_test_out.out grb_test_out.out
# echo "data, K=128-cusparse-gflops, K=128-gespmm-gflops,K=256-cusparse-gflops,K=256-gespmm-gflops,K=512-cusparse-gflops,K=512-gespmm-gflops," >> spmm_test_out.out
cd ./ge-spmm

# rm -f gespmm_test.log
# make 

run_gespmm(){
    data=$1
    N=$2
    ./gespmm_test $data $device $N
}

DIR="/home/ubuntu/gnn_benchmark/spmm_test/datasets"

# echo "graph, n columns, kernel time, kernel gflops, ..."

# graph="/home/ubuntu/gnn_benchmark/spmm_test/datasets/pubmed.mtx"
# echo "N = 32"
# run_gespmm $graph 32
# echo "N = 3"
# run_gespmm $graph 3

# graph="/home/ubuntu/gnn_benchmark/spmm_test/datasets/ogbn-arxiv.mtx"
# echo "N = 128"
# run_gespmm $graph 128
# echo "N = 256"
# run_gespmm $graph 256
# echo "N = 40"
# run_gespmm $graph 40

# graph="/home/ubuntu/gnn_benchmark/spmm_test/datasets/ogbn-proteins.mtx"
# echo "N = 8"
# run_gespmm $graph 8
# echo "N = 256"
# run_gespmm $graph 256
# echo "N = 112"
# run_gespmm $graph 112

graph="$DIR/reddit_s64.mtx"
echo "N = 128"
run_gespmm $graph 128
echo "N = 41"
run_gespmm $graph 41

# graph="$DIR/reddit_s128.mtx"
# echo "N = 128"
# run_gespmm $graph 128
# echo "N = 41"
# run_gespmm $graph 41
# 
# graph="$DIR/reddit_s256.mtx"
# echo "N = 128"
# run_gespmm $graph 128
# echo "N = 41"
# run_gespmm $graph 41

cat gespmm_test.log

cd ..
