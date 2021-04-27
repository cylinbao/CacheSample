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

echo "Testing for GCN Settings"
echo "Testinig for S = 32"
graph="/home/ubuntu/gnn_benchmark/spmm_test/datasets/pubmed.mtx"
echo $graph
echo "N = 32, S = 32"
run_csspmm $graph 500 32
echo "N = 32, S = 32"
run_csspmm $graph 16 32

graph="/home/ubuntu/gnn_benchmark/spmm_test/datasets/ogbn-arxiv.mtx"
echo $graph
echo "N = 128, S = 32"
run_csspmm $graph 128 32
echo "N = 256, S = 32"
run_csspmm $graph 256 32
echo "N = 40, S = 32"
run_csspmm $graph 256 32

graph="/home/ubuntu/gnn_benchmark/spmm_test/datasets/ogbn-proteins.mtx"
echo $graph
echo "N = 8, S = 32"
run_csspmm $graph 8 32
echo "N = 256, S = 32"
run_csspmm $graph 256 32
echo "N = 112, S = 32"
run_csspmm $graph 256 32

graph="/home/ubuntu/gnn_benchmark/spmm_test/datasets/reddit.mtx"
echo $graph
echo "N = 128, S = 32"
run_csspmm $graph 602 32
echo "N = 41, S = 32"
run_csspmm $graph 128 32

echo "Testing for GCN Settings"
echo "Testinig for S = 128"
graph="/home/ubuntu/gnn_benchmark/spmm_test/datasets/pubmed.mtx"
echo $graph
echo "N = 32, S = 128"
run_csspmm $graph 500 128
echo "N = 32, S = 128"
run_csspmm $graph 16 128

graph="/home/ubuntu/gnn_benchmark/spmm_test/datasets/ogbn-arxiv.mtx"
echo $graph
echo "N = 128, S = 128"
run_csspmm $graph 128 128
echo "N = 256, S = 128"
run_csspmm $graph 256 128
echo "N = 40, S = 128"
run_csspmm $graph 256 128

graph="/home/ubuntu/gnn_benchmark/spmm_test/datasets/ogbn-proteins.mtx"
echo $graph
echo "N = 8, S = 128"
run_csspmm $graph 8 128
echo "N = 256, S = 128"
run_csspmm $graph 256 128
echo "N = 112, S = 128"
run_csspmm $graph 256 128

graph="/home/ubuntu/gnn_benchmark/spmm_test/datasets/reddit.mtx"
echo $graph
echo "N = 128, S = 128"
run_csspmm $graph 602 128
echo "N = 41, S = 128"
run_csspmm $graph 128 128

cat csspmm_test.log

cd ..
