# echo filename, num_rows, num_cols, num_nonzeros, row_length_mean, row_length_std_dev, row_length_variation, row_length_skewness, small, big, method1, avg_ms, gflops, gbps, method2, avg_ms, gflops, gbps, method3, avg_ms, gflops, gbps,
# for file in /data/ctcyang/GraphBLAS/dataset/europar/highd/*/
# for data in ./datasets/*

cd merge-spmm

LOG_FILE=mgspmm_test_gcn_sample.log

# rm -f $LOG_FILE

run_gbspmm(){
    N=$1
    data=$2
    ./bin/gbspmm --max_ncols=$N --iter=200 --device=0 $data >> $LOG_FILE
}

echo "filename, num_rows, num_cols, num_nonzeros, row_length_mean, row_length_std_dev, row_length_variation, row_length_skewness, small, big, method1, avg_ms, gflops, gbps, method2, avg_ms, gflops, gbps, method3, avg_ms, gflops, gbps," >> $LOG_FILE 

DIR="/home/ubuntu/gnn_benchmark/spmm_test/datasets/"

# echo "Processing Pubmed"
# graph="/home/ubuntu/gnn_benchmark/spmm_test/datasets/pubmed.mtx"
# echo "n_cols, 32" >> $LOG_FILE 
# run_gbspmm 32 $graph
# echo "n_cols, 3" >> $LOG_FILE
# run_gbspmm 3 $graph
# 
# echo "Processing Arxiv"
# graph="/home/ubuntu/gnn_benchmark/spmm_test/datasets/ogbn-arxiv.mtx"
# echo "n_cols, 128" >> $LOG_FILE
# run_gbspmm 128 $graph
# echo "n_cols, 256" >> $LOG_FILE
# run_gbspmm 256 $graph
# echo "n_cols, 40" >> $LOG_FILE
# run_gbspmm 40 $graph
# 
# echo "Processing Proteins"
# graph="/home/ubuntu/gnn_benchmark/spmm_test/datasets/ogbn-proteins.mtx"
# echo "n_cols, 8" >> $LOG_FILE
# # ./bin/gbspmm --max_ncols=8 --iter=50 --device=0 $graph >> $LOG_FILE
# run_gbspmm 8 $graph
# echo "n_cols, 256" >> $LOG_FILE
# run_gbspmm 256 $graph
# echo "n_cols, 112" >> $LOG_FILE
# run_gbspmm 112 $graph

echo "Processing Reddit"
graph="$DIR/reddit_s64.mtx"
echo "n_cols, 128" >> $LOG_FILE
run_gbspmm 128 $graph
echo "n_cols, 41" >> $LOG_FILE
run_gbspmm 41 $graph

# echo "Processing Reddit"
# graph="$DIR/reddit_s128.mtx"
# echo "n_cols, 128" >> $LOG_FILE
# run_gbspmm 128 $graph
# echo "n_cols, 41" >> $LOG_FILE
# run_gbspmm 41 $graph
# 
# echo "Processing Reddit"
# graph="$DIR/reddit_s256.mtx"
# echo "n_cols, 128" >> $LOG_FILE
# run_gbspmm 128 $graph
# echo "n_cols, 41" >> $LOG_FILE
# run_gbspmm 41 $graph

cat $LOG_FILE

cd ..
