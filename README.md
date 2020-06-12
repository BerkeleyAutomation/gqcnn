
## Berkeley AUTOLAB's GQCNN Package On Web Services

### Setup
- Basic setup
```
sudo apt install python3-rtree
pip3 install .
```
- Web setup
```
pip3 install -r requirements/web_requirements.txt
pip3 install -r requirements/linting_requirements.txt
```
- Download the model and examples in the gqcnn website

### Run
- Start web app
```
python3 main.py
```
- Build and run in docker
```
docker-compose up --build
```
- Run in docker
```
docker-compose up
```

### Reference
- https://berkeleyautomation.github.io/gqcnn/
