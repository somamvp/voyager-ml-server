import redis

# 레디스 연결
rd: redis.StrictRedis = redis.StrictRedis(host="localhost", port=6379, db=0)

# 레디스에서 키를 사용해서 hash 값 가져오기
# id = session_id
# key = "yolo:" + id
# data = rd.hget(key, "yoloResult")

# print(data)
