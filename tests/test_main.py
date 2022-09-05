import json
import pytest

from httpx import AsyncClient

# @pytest.mark.asyncio
# async def test_root():
#     async with AsyncClient(base_url="http://43.200.197.251/fast") as ac:
#         response = await ac.get("/")
#         assert response.status_code == 200
#         assert response.json() == {"message": "Hello World"}

# 이제 세가지 API 함수에 대해 테스트하는 함수들을 작성하겠습니다.

# test_root(): “root“ API 함수에 대해 GET 방식으로 요청하여 테스트하는 함수 입니다.

# test_read_item(): “read_item“ API 함수에 대해 item_id가 1인 item을 params 값에 추가하여 GET 방식으로 요청하여 테스트하는 함수 입니다.

# test_create_item(): “create_item“ API 함수에 대해 생성할 item을 json 값에 추가하여 POST 방식으로 요청하여 테스트하는 함수 입니다.

@pytest.mark.asyncio
async def test_read_item():
    async with AsyncClient(base_url="http://127.0.0.1:8000") as ac:
        response = await ac.get("/items/foo", params={"item_id": "1"})
        assert response.status_code == 200
        assert response.json() == {
            "id": "foo",
            "title": "Foo",
            "description": "There goes my hero",
        }


@pytest.mark.asyncio
async def test_create_item():
    async with AsyncClient(base_url="http://127.0.0.1:8000") as ac:
        response = await ac.post(
            "/items/",
            content=json.dumps(
                {
                    "id": "foobar",
                    "title": "Foo Bar",
                    "description": "The Foo Barters",
                }
            ),
        )
        assert response.status_code == 200
        assert response.json() == {
            "id": "foobar",
            "title": "Foo Bar",
            "description": "The Foo Barters",
        }