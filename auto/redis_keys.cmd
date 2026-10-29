:: auto/redis_keys.cmd
:: Purpose: List all estimation keys stored in the estimator-redis Docker container
:: Usage: auto\redis_keys.cmd (run from the project root directory)

docker exec -it estimator-redis redis-cli keys "estimation:*"