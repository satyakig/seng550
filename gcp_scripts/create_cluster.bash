#!/bin/bash
# ./create_cluster.bash [cluster-name]
# ./create_cluster.bash seng550
# https://cloud.google.com/compute/docs/machine-types#custom_machine_types
# https://spark.apache.org/docs/latest/configuration.html

NAME=$1

gcloud dataproc clusters create ${NAME} \
  --scopes=cloud-platform \
  --region=us-central1 \
  --zone=us-central1-a \
  --bucket=pyspark_bucket \
  --max-idle=1h \
  --image-version=1.4 \
  --optional-components=ANACONDA,JUPYTER \
  --num-masters=1 \
  --num-workers=4 \
  --master-machine-type=n1-highmem-16 \
  --master-boot-disk-type=pd-ssd \
  --master-boot-disk-size=400GB \
  --worker-machine-type=n1-highmem-16 \
  --worker-boot-disk-size=400GB \
  --worker-boot-disk-type=pd-ssd \
  --preemptible-worker-accelerator type=nvidia-tesla-k80,count=4 \
  --initialization-actions gs://dataproc-initialization-actions/python/conda-install.sh,gs://dataproc-initialization-actions/python/pip-install.sh,gs://seng550/bash_scripts/install_gpu_driver.bash \
  --metadata 'CONDA_PACKAGES=tensorflow keras' \
  --metadata 'PIP_PACKAGES=numpy==1.17.3 pandas==0.25.3 scipy==1.3.2 elephas systemml sklearn'