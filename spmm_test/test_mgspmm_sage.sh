# echo filename, num_rows, num_cols, num_nonzeros, row_length_mean, row_length_std_dev, row_length_variation, row_length_skewness, small, big, method1, avg_ms, gflops, gbps, method2, avg_ms, gflops, gbps, method3, avg_ms, gflops, gbps,
# for file in /data/ctcyang/GraphBLAS/dataset/europar/highd/*/
# for data in ./datasets/*

cd merge-spmm

LOG_FILE=mgspmm_test_sage.log

# rm -f $LOG_FILE

run_gbspmm(){
    N=$1
    data=$2
    ./bin/gbspmm --max_ncols=$N --iter=50 --device=0 $data >> $LOG_FILE
}

echo "filename, num_rows, num_cols, num_nonzeros, row_length_mean, row_length_std_dev, row_length_variation, row_length_skewness, small, big, method1, avg_ms, gflops, gbps, method2, avg_ms, gflops, gbps, method3, avg_ms, gflops, gbps," >> $LOG_FILE 

echo "Processing Pubmed"
graph="/home/ubuntu/gnn_benchmark/spmm_test/datasets/pubmed.mtx"
echo "n_cols, 500" >> $LOG_FILE 
run_gbspmm 500 $graph
echo "n_cols, 16" >> $LOG_FILE
run_gbspmm 16 $graph

echo "Processing Arxiv"
graph="/home/ubuntu/gnn_benchmark/spmm_test/datasets/ogbn-arxiv.mtx"
echo "n_cols, 128" >> $LOG_FILE
run_gbspmm 128 $graph
echo "n_cols, 256" >> $LOG_FILE
run_gbspmm 256 $graph
echo "n_cols, 256" >> $LOG_FILE
run_gbspmm 256 $graph

echo "Processing Proteins"
graph="/home/ubuntu/gnn_benchmark/spmm_test/datasets/ogbn-proteins.mtx"
echo "n_cols, 8" >> $LOG_FILE
run_gbspmm 8 $graph
echo "n_cols, 256" >> $LOG_FILE
run_gbspmm 256 $graph
echo "n_cols, 256" >> $LOG_FILE
run_gbspmm 256 $graph

echo "Processing Reddit"
graph="/home/ubuntu/gnn_benchmark/spmm_test/datasets/reddit.mtx"
echo "n_cols, 602" >> $LOG_FILE
run_gbspmm 602 $graph
echo "n_cols, 128" >> $LOG_FILE
run_gbspmm 128 $graph

cat $LOG_FILE

cd ..
