# POC opus-mt sentence split neural machine translation

## Convert model to onnx

´´´
git clone <https://huggingface.co/jkorsvik/opus-mt-eng-nor> ./models/en-no
´´´

## Build docker

´´´
docker build -t torch_onnx_cuda_runtime .
´´´
