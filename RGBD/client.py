import requests

files = open('sample_D.jpg', 'rb')

upload = {'source': files}

res = requests.post('http://127.0.0.1:9000/upload', files = upload)

print(res.content)