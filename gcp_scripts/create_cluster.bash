#!/bin/bash
# ./create_cluster.bash [cluster-name]
# ./create_cluster.bash seng550

NAME=$1

gcloud dataproc clusters create ${NAME} \
  --scopes=cloud-platform \
  --region=us-central1 \
  --bucket=pyspark_bucket \
  --max-idle=1h \
  --image-version=1.4 \
  --optional-components=ANACONDA,JUPYTER \
  --num-masters=1 \
  --num-workers=5 \
  --master-machine-type=n1-highmem-4 \
  --master-boot-disk-type=pd-ssd \
  --master-boot-disk-size=100GB \
  --worker-machine-type=n1-highmem-4 \
  --worker-boot-disk-size=80GB \
  --worker-boot-disk-type=pd-ssd \
  --initialization-actions gs://dataproc-initialization-actions/python/conda-install.sh,gs://dataproc-initialization-actions/python/pip-install.sh \
  --metadata 'CONDA_PACKAGES=tensorflow keras sklearn' \
  --metadata 'PIP_PACKAGES=numpy==1.17.3 pandas==0.25.3 scipy==1.3.2 elephas systemml'
