## Construye el contenedor localmente
build:
	export PROJECT_ID=$(gcloud config list --format 'value(core.project)')
	docker build -t gcr.io/${PROJECT_ID}/forex_update:latest .

## Corre el contenedor hecho en build
run:
	docker run -d forex_update:latest
## Hace push el contenedor creado hacia el repositorio que esté configurado
push:
	docker build -t gcr.io/${PROJECT_ID}/forex_update:latest .
	gcloud docker -- push gcr.io/${PROJECT_ID}/forex_update:latest
## Elimina lo creado por build y run
clean:
	@bash -x scripts/uninstall.sh
