  set -euo pipefail

  docker-compose up -d

  docker exec -u root spark-master chmod -R 777 /usr/local/spark/
  docker exec -u root spark-worker-1 chmod -R 777 /usr/local/spark/
  docker exec -u root spark-worker-2 chmod -R 777 /usr/local/spark/

  docker exec spark-master /usr/local/spark/sbin/start-master.sh
  docker exec spark-worker-1 /usr/local/spark/sbin/start-worker.sh spark://spark-master:7077
  docker exec spark-worker-2 /usr/local/spark/sbin/start-worker.sh spark://spark-master:7077

  echo "Cluster listo."