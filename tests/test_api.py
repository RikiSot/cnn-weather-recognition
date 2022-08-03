from fastapi.testclient import TestClient
from fastapi import status
from api import app


class TestAPI:
    client = TestClient(app)

    def test_home(self):
        r = self.client.get('/')
        assert r.status_code == 200
        assert r.json() == {'message': 'Technical test'}

    def test_prediction(self):
        img = {'image_link': 'https://www.dtn.com/wp-content/uploads/2019/03/hard_rime.jpg'}
        r = self.client.post(url='/predict', params=img)
        assert r.status_code == 200
        assert r.json()['label'] == 'rime'

