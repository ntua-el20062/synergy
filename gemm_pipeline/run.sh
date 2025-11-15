export CUBLAS_WORKSPACE_CONFIG=:16:8

for N in 2000 20000; do

echo "batch size = 5, num steps = 10"
./gemm_pipeline $N 5 10
echo "-----------------------"
./gemm_pipeline $N 5 10
echo "-----------------------"
./gemm_pipeline $N 5 10
echo "==============================================="

echo "batch size = 10, num steps = 20"
./gemm_pipeline $N 10 20
echo "-----------------------"
./gemm_pipeline $N 10 20
echo "-----------------------"
./gemm_pipeline $N 10 20
echo "==============================================="


echo "batch size = 5, num steps = 15"
./gemm_pipeline $N 5 15
echo "-----------------------"
./gemm_pipeline $N 5 15
echo "-----------------------"
./gemm_pipeline $N 5 15
echo "==============================================="

echo "batch size = 50, num steps = 100"
./gemm_pipeline $N 50 100
echo "-----------------------"
./gemm_pipeline $N 50 100
echo "-----------------------"
./gemm_pipeline $N 50 100
echo "==============================================="

echo "batch size = 20, num steps = 100"
./gemm_pipeline $N 20 100
echo "-----------------------"
./gemm_pipeline $N 20 100
echo "-----------------------"
./gemm_pipeline $N 20 100
echo "==============================================="

echo "batch size = 75, num steps = 150"
./gemm_pipeline $N 75 150
echo "-----------------------"
./gemm_pipeline $N 75 150
echo "-----------------------"
./gemm_pipeline $N 75 150
echo "==============================================="

echo "batch size = 50, num steps = 150"
./gemm_pipeline $N 50 150
echo "-----------------------"
./gemm_pipeline $N 50 150
echo "-----------------------"
./gemm_pipeline $N 50 150
echo "==============================================="
done
