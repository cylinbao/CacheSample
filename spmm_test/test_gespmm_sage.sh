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

# echo "graph, n columns, kernel time, kernel gflops, ..."

graph="/home/ubuntu/gnn_benchmark/spmm_test/datasets/pubmed.mtx"
echo "N = 500"
run_gespmm $graph 500
echo "N = 16"
run_gespmm $graph 16

graph="/home/ubuntu/gnn_benchmark/spmm_test/datasets/ogbn-arxiv.mtx"
echo "N = 128"
run_gespmm $graph 128
echo "N = 256"
run_gespmm $graph 256
echo "N = 256"
run_gespmm $graph 256

graph="/home/ubuntu/gnn_benchmark/spmm_test/datasets/ogbn-proteins.mtx"
echo "N = 8"
run_gespmm $graph 8
echo "N = 256"
run_gespmm $graph 256
echo "N = 256"
run_gespmm $graph 256

graph="/home/ubuntu/gnn_benchmark/spmm_test/datasets/reddit.mtx"
echo "N = 602"
run_gespmm $graph 602
echo "N = 128"
run_gespmm $graph 128

cat gespmm_test.log

cd ..
