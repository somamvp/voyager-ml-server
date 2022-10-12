import redis

# 레디스 연결
rd = redis.StrictRedis(host='localhost', port=6379, db=0)

# 레디스에서 키를 사용해서 값 가져오기
rd.set("name", "Jhon")
a = rd.get("name")
print(a)

string = sm.serialize()
rd.get("sdlkfj-sdklfj-12312sakdf-sdfasd")
