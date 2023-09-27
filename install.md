### Installation tutorial 
--------------------------------------------------------------------------------------------------------------------------------
#### Ubuntu
- install anaconda : https://docs.anaconda.com/free/anaconda/install/linux/
- install OpenFace : https://github.com/TadasBaltrusaitis/OpenFace/wiki/Unix-Installation
    - for Ubuntu < 22.04
        1. Download both files from the github repository : download_models.sh and install.sh
        2. installation
            ```
            ./download_models.sh
            ./install.sh
            ```
        3. testing
            ```
            ./bin/FaceLandmarkVid -f "../samples/changeLighting.wmv" -f "../samples/2015-10-15-15-14.avi"
            ```
    - Using Docker
        - install Docker
        - run the docker script
            ```
            docker run -it --rm algebr/openface:latest
            ```
- activate environment and install requirements
    1. clone the repository
        ```
        git clone "AU_BS_model"
        cd AU_BS_model
        ```
    2. create environment and install requirements
        ```
        conda create --name grunet_env --file requirements.txt --channel default --channel anaconda
        conda activate grunet_env
        ```
    3. test with our demo
        ```
        demo.py "path_to_video" "path_to_result"
        ```