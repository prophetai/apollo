bootstrap: download_models

download_models: install_wkhtmltopdf
	gsutil cp 'gs://forex_models/models/gb*'  apollo/src/assets/models_1h/
    gsutil cp -r 'gs://forex_models/models/variables'  apollo/src/assets/models_1h/

    gsutil cp -r 'gs://forex_models/models_4h'  apollo/src/assets/models_4h/

install_wkhtmltopdf: install_requirements
	wget https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6-1/wkhtmltox_0.12.6-1.bionic_amd64.deb && \
        sudo apt-get -y install xfonts-75dpi && \
        sudo apt-get -y install xfonts-base && \
        sudo dpkg -i wkhtmltox_0.12.6-1.bionic_amd64.deb && \
        sudo apt-get install -f

install_requirements:
	pip3 install -r requirements.txt 


run.script:
	python3 src/main.py -o -t --model-version=models_1h --instrument=USD_JPY --account=1h